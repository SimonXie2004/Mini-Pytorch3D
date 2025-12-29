#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <mini_pytorch3d/constants.h>

namespace mini_pytorch3d {
    
    __device__ __forceinline__
    int clamp_int(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    __device__ __forceinline__
    void tri_screen_bbox(
        const float* clip_pos,
        const float* inv_w,
        const int64_t* faces,
        int f,
        int W, int H,
        int& xmin, int& xmax,
        int& ymin, int& ymax
    ) {
        int i0 = (int)faces[3*f + 0];
        int i1 = (int)faces[3*f + 1];
        int i2 = (int)faces[3*f + 2];

        float w0 = inv_w[i0];
        float w1 = inv_w[i1];
        float w2 = inv_w[i2];

        float x0 = clip_pos[4*i0 + 0] * w0;
        float y0 = clip_pos[4*i0 + 1] * w0;
        float x1 = clip_pos[4*i1 + 0] * w1;
        float y1 = clip_pos[4*i1 + 1] * w1;
        float x2 = clip_pos[4*i2 + 0] * w2;
        float y2 = clip_pos[4*i2 + 1] * w2;

        float sx0 = (x0 * 0.5f + 0.5f) * W;
        float sy0 = (-y0 * 0.5f + 0.5f) * H;
        float sx1 = (x1 * 0.5f + 0.5f) * W;
        float sy1 = (-y1 * 0.5f + 0.5f) * H;
        float sx2 = (x2 * 0.5f + 0.5f) * W;
        float sy2 = (-y2 * 0.5f + 0.5f) * H;

        float minx = fminf(sx0, fminf(sx1, sx2));
        float maxx = fmaxf(sx0, fmaxf(sx1, sx2));
        float miny = fminf(sy0, fminf(sy1, sy2));
        float maxy = fmaxf(sy0, fmaxf(sy1, sy2));

        xmin = clamp_int((int)floorf(minx), 0, W - 1);
        xmax = clamp_int((int)ceilf (maxx), 0, W - 1);
        ymin = clamp_int((int)floorf(miny), 0, H - 1);
        ymax = clamp_int((int)ceilf (maxy), 0, H - 1);
    }

    // pass 1: count how many triangles per tile
    __global__ void binning_count_kernel(
        const float* clip_pos,
        const float* inv_w,
        const int64_t* faces,
        int F,
        int W, int H,
        int tiles_x, int tiles_y,
        int* tile_counts
    ) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= F) return;

        int xmin, xmax, ymin, ymax;
        tri_screen_bbox(clip_pos, inv_w, faces, f, W, H, xmin, xmax, ymin, ymax);

        int tx0 = xmin / TILE;
        int tx1 = xmax / TILE;
        int ty0 = ymin / TILE;
        int ty1 = ymax / TILE;

        tx0 = clamp_int(tx0, 0, tiles_x - 1);
        tx1 = clamp_int(tx1, 0, tiles_x - 1);
        ty0 = clamp_int(ty0, 0, tiles_y - 1);
        ty1 = clamp_int(ty1, 0, tiles_y - 1);

        for (int ty = ty0; ty <= ty1; ++ty) {
            int base = ty * tiles_x;
            for (int tx = tx0; tx <= tx1; ++tx) {
                atomicAdd(&tile_counts[base + tx], 1);
            }
        }
    }

    // pass 2: write triangle ids
    __global__ void binning_fill_kernel(
        const float* clip_pos,
        const float* inv_w,
        const int64_t* faces,
        int F,
        int W, int H,
        int tiles_x, int tiles_y,
        const int* tile_offsets,   // [num_tiles]
        int* tile_write,           // [num_tiles]
        int* tile_faces_flat       // [total_pairs]
    ) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= F) return;

        int xmin, xmax, ymin, ymax;
        tri_screen_bbox(clip_pos, inv_w, faces, f, W, H, xmin, xmax, ymin, ymax);

        int tx0 = xmin / TILE;
        int tx1 = xmax / TILE;
        int ty0 = ymin / TILE;
        int ty1 = ymax / TILE;

        tx0 = clamp_int(tx0, 0, tiles_x - 1);
        tx1 = clamp_int(tx1, 0, tiles_x - 1);
        ty0 = clamp_int(ty0, 0, tiles_y - 1);
        ty1 = clamp_int(ty1, 0, tiles_y - 1);

        for (int ty = ty0; ty <= ty1; ++ty) {
            int base = ty * tiles_x;
            for (int tx = tx0; tx <= tx1; ++tx) {
                int t = base + tx;
                int idx = atomicAdd(&tile_write[t], 1);
                tile_faces_flat[idx] = (int)f;
            }
        }
    }
    
    void launch_binning_two_pass(
        const float* clip_pos,
        const float* inv_w,
        const int64_t* faces,
        int F,
        int W, int H,
        int* tile_offsets,     // [num_tiles+1] (output)
        int* tile_faces_flat,  // [total_pairs] (output)
        int* total_pairs_out,  // [1] device scalar (output)
        cudaStream_t stream
    ) {
        int tiles_x = (W + TILE - 1) / TILE;
        int tiles_y = (H + TILE - 1) / TILE;
        int num_tiles = tiles_x * tiles_y;

        int* tile_counts;
        cudaMallocAsync(&tile_counts, num_tiles * sizeof(int), stream);
        cudaMemsetAsync(tile_counts, 0, num_tiles * sizeof(int), stream);

        // pass 1: count
        int threads = 256;
        int blocks  = (F + threads - 1) / threads;

        binning_count_kernel<<<blocks, threads, 0, stream>>>(
            clip_pos, inv_w, faces,
            F, W, H,
            tiles_x, tiles_y,
            tile_counts
        );

        // prefix sum (counts -> offsets)
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes,
            tile_counts,
            tile_offsets,
            num_tiles,
            stream
        );

        void* cub_temp;
        cudaMallocAsync(&cub_temp, temp_bytes, stream);

        cub::DeviceScan::ExclusiveSum(
            cub_temp, temp_bytes,
            tile_counts,
            tile_offsets,
            num_tiles,
            stream
        );

        // last offset = total_pairs
        cudaMemcpyAsync(
            total_pairs_out,
            tile_offsets + (num_tiles - 1),
            sizeof(int),
            cudaMemcpyDeviceToDevice,
            stream
        );

        // tile_write = tile_offsets (reuse tile_counts buffer)
        cudaMemcpyAsync(
            tile_counts,
            tile_offsets,
            num_tiles * sizeof(int),
            cudaMemcpyDeviceToDevice,
            stream
        );

        // pass 2: fill
        binning_fill_kernel<<<blocks, threads, 0, stream>>>(
            clip_pos, inv_w, faces,
            F, W, H,
            tiles_x, tiles_y,
            tile_offsets,
            tile_counts,
            tile_faces_flat
        );

        cudaFreeAsync(tile_counts, stream);
        cudaFreeAsync(cub_temp, stream);
    }

} // namespace mini_pytorch3d