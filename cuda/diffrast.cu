#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mini_pytorch3d/constants.h>

namespace mini_pytorch3d {

    __device__ __forceinline__
    float edge_fn(float ax, float ay, float bx, float by, float px, float py) {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    }

    __device__ __forceinline__
    bool is_top_left(float ax, float ay, float bx, float by) {
        float dy = by - ay;
        float dx = bx - ax;
        return (dy > 0) || (dy == 0 && dx < 0);
    }

    __device__ __forceinline__
    float3 f3_add(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ __forceinline__
    float3 f3_mul(const float3& a, float s) {
        return make_float3(a.x * s, a.y * s, a.z * s);
    }

    __device__ __forceinline__
    float f3_dot(const float3& a, const float3& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    __device__ __forceinline__
    float3 f3_normalize(const float3& v) {
        float n2 = f3_dot(v, v);
        float inv = rsqrtf(fmaxf(n2, 1e-20f));
        return f3_mul(v, inv);
    }

    __device__ __forceinline__
    float3 diffuse_only_phong_shading(
        const float3& normal,
        const float3& light_dir,
        const float3& base_color
    ) {
        // double-sided diffuse
        float ndl1 = f3_dot(normal, light_dir);
        float ndl2 = f3_dot(f3_mul(normal, -1.f), light_dir);
        float diffuse = fmaxf(0.f, fmaxf(ndl1, ndl2));

        return f3_mul(base_color, diffuse);
    }

    __global__ void parallel_diffrast_kernel(
        const float* clip_pos,
        const float* inv_w,
        const float* colors,
        const float* normals,
        const int64_t* faces,
        const int32_t* tile_offsets,
        const int32_t* tile_faces,
        int W, int H, float sigma,
        const float* light_dir,
        float* out
    ) {
        int tx = blockIdx.x;
        int ty = blockIdx.y;

        int lx = threadIdx.x;
        int ly = threadIdx.y;
        int tid = ly * TILE + lx;

        int px = tx * TILE + lx;
        int py = ty * TILE + ly;

        __shared__ float sh_max[TILE_PIX];
        __shared__ float sh_sum[TILE_PIX];
        __shared__ float3 sh_sum_c[TILE_PIX];

        sh_max[tid] = -1e10f;
        sh_sum[tid] = 0.f;
        sh_sum_c[tid] = make_float3(0,0,0);
        __syncthreads();

        int tile_id = ty * gridDim.x + tx;
        int begin = tile_offsets[tile_id];
        int end = tile_offsets[tile_id + 1];

        float px_c = px + 0.5f;
        float py_c = py + 0.5f;

        for (int it = begin; it < end; ++it) {
            int f = tile_faces[it];
            int i0 = faces[3*f+0];
            int i1 = faces[3*f+1];
            int i2 = faces[3*f+2];

            float w0 = inv_w[i0];
            float w1 = inv_w[i1];
            float w2 = inv_w[i2];

            float x0 = clip_pos[4*i0+0] * w0;
            float y0 = clip_pos[4*i0+1] * w0;
            float z0 = clip_pos[4*i0+2] * w0;
            float x1 = clip_pos[4*i1+0] * w1;
            float y1 = clip_pos[4*i1+1] * w1;
            float z1 = clip_pos[4*i1+2] * w1;
            float x2 = clip_pos[4*i2+0] * w2;
            float y2 = clip_pos[4*i2+1] * w2;
            float z2 = clip_pos[4*i2+2] * w2;

            float sx0 = (x0 * 0.5f + 0.5f) * W;
            float sy0 = (-y0 * 0.5f + 0.5f) * H;
            float sx1 = (x1 * 0.5f + 0.5f) * W;
            float sy1 = (-y1 * 0.5f + 0.5f) * H;
            float sx2 = (x2 * 0.5f + 0.5f) * W;
            float sy2 = (-y2 * 0.5f + 0.5f) * H;

            float area = edge_fn(sx0, sy0, sx1, sy1, sx2, sy2);
            if (fabsf(area) < 1e-8f) continue;

            float e0 = edge_fn(sx1, sy1, sx2, sy2, px_c, py_c);
            float e1 = edge_fn(sx2, sy2, sx0, sy0, px_c, py_c);
            float e2 = edge_fn(sx0, sy0, sx1, sy1, px_c, py_c);

            bool inside =
                (e0 > 0 || (e0 == 0 && is_top_left(sx1, sy1, sx2, sy2))) &&
                (e1 > 0 || (e1 == 0 && is_top_left(sx2, sy2, sx0, sy0))) &&
                (e2 > 0 || (e2 == 0 && is_top_left(sx0, sy0, sx1, sy1)));

            if (!inside) continue;

            float b0 = e0 / area;
            float b1 = e1 / area;
            float b2 = 1.f - b0 - b1;
            if (b0 < 0 || b1 < 0 || b2 < 0) continue;

            float depth = b0*z0 + b1*z1 + b2*z2;

            // diffuse only shading
            float3 n = make_float3(
                normals[3*i0+0]*b0 + normals[3*i1+0]*b1 + normals[3*i2+0]*b2,
                normals[3*i0+1]*b0 + normals[3*i1+1]*b1 + normals[3*i2+1]*b2,
                normals[3*i0+2]*b0 + normals[3*i1+2]*b1 + normals[3*i2+2]*b2
            );
            float3 nn = f3_normalize(n);
            float3 L = make_float3(light_dir[0], light_dir[1], light_dir[2]);
            float3 base = make_float3(
                colors[3*i0+0]*b0 + colors[3*i1+0]*b1 + colors[3*i2+0]*b2,
                colors[3*i0+1]*b0 + colors[3*i1+1]*b1 + colors[3*i2+1]*b2,
                colors[3*i0+2]*b0 + colors[3*i1+2]*b1 + colors[3*i2+2]*b2
            );
            float3 col = diffuse_only_phong_shading(nn, L, base);

            // online softmax
            float logit = -depth / sigma;

            float old_max = sh_max[tid];
            float new_max = fmaxf(old_max, logit);
            float rescale = expf(old_max - new_max);
            float exp_new = expf(logit - new_max);

            sh_sum[tid] = sh_sum[tid] * rescale + exp_new;
            sh_sum_c[tid] = f3_add(
                f3_mul(sh_sum_c[tid], rescale), 
                f3_mul(col, exp_new)
            );
            sh_max[tid] = new_max;
        }

        if (px < W && py < H && sh_sum[tid] > 0) {
            float3 c = f3_mul(sh_sum_c[tid], 1.f / sh_sum[tid]);
            int p = (py * W + px) * 3;
            out[p+0] = c.x;
            out[p+1] = c.y;
            out[p+2] = c.z;
        }

    }

    void launch_parallel_diffrast_kernel(
        const float* clip_pos,
        const float* inv_w,
        const float* colors,
        const float* normals,
        const int64_t* faces,
        const int32_t* tile_offsets,
        const int32_t* tile_faces,
        int W, int H, float sigma,
        const float* light_dir,
        float* out
    ) {
        dim3 block(TILE, TILE);
        dim3 grid((W + TILE - 1) / TILE,
                (H + TILE - 1) / TILE);

        parallel_diffrast_kernel<<<grid, block>>>(
            clip_pos,
            inv_w,
            colors,
            normals,
            faces,
            tile_offsets,
            tile_faces,
            W, H, sigma,
            light_dir,
            out
        );
    }

} // namespace mini_pytorch3d
