import torch
import time
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)
from PIL import Image
import numpy as np

def test_rendering_speed(obj_path, num_iterations=100, image_size=512, sigma=1e-4, gamma=1e-4, output_path=None):
    """
    Test the rendering speed of a 3D model using PyTorch3D
    
    Args:
        obj_path: Path to the OBJ file
        num_iterations: Number of rendering iterations for timing
        image_size: Size of the rendered image (height and width)
        sigma: Sigma parameter for soft rasterization (controls the sharpness of edges)
        gamma: Gamma parameter for blending (controls the transparency blending)
        output_path: Path to save the rendered image (optional)
    
    Returns:
        Average rendering time in milliseconds
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"\nLoading model from: {obj_path}")
    
    # Load the mesh
    mesh = load_objs_as_meshes([obj_path], device=device)
    
    # Get mesh statistics
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    print(f"Mesh loaded - Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}")
    
    # Always use vertex colors based on normalized vertex positions
    # Normalize vertex positions to [0, 1] range for RGB colors
    verts_min = verts.min(dim=0)[0]
    verts_max = verts.max(dim=0)[0]
    verts_normalized = (verts - verts_min) / (verts_max - verts_min + 1e-8)
    verts_rgb = verts_normalized[None]  # (1, V, 3) - use normalized positions as RGB
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh.textures = textures
    
    # Setup camera
    R, T = look_at_view_transform(dist=1, elev=0, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Setup rasterization settings with sigma for soft rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=10,  # Allow multiple faces per pixel for soft blending
    )
    
    # Setup lights
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    
    # Setup blend parameters with gamma
    blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(0.0, 0.0, 0.0))
    
    # Create renderer with differentiable soft rasterization
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )
    
    # # Warm-up renders (to avoid including initialization time)
    # print(f"\nWarming up with 10 renders...")
    # for _ in range(10):
    #     with torch.no_grad():
    #         images = renderer(mesh)
    
    # # Synchronize CUDA if using GPU
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    
    # Timed rendering
    print(f"Running {num_iterations} timed renders...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            images = renderer(mesh)
    
    # Synchronize CUDA if using GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate statistics
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_iterations
    fps = 1000 / avg_time_ms
    
    # Print results
    print("\n" + "="*60)
    print("RENDERING PERFORMANCE RESULTS")
    print("="*60)
    print(f"Model: {obj_path}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Sigma: {sigma}, Gamma: {gamma}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Average rendering time: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("="*60)
    
    # Save the rendered image if output_path is provided
    if output_path is not None:
        # images is (1, H, W, 4) with RGBA channels, convert to RGB
        image_np = images[0, ..., :3].cpu().numpy()  # Take first image, RGB channels
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)  # Convert to 0-255 range
        
        # Save using PIL
        img = Image.fromarray(image_np)
        img.save(output_path)
        print(f"\nSaved rendered image to: {output_path}")
    
    return avg_time_ms


if __name__ == "__main__":
    obj_path = "data/utah_teapot.obj"
    
    # Test with different image sizes
    print("Testing rendering speed for Stanford Dragon model\n")
    
    # Test with 512x512 resolution and diffrast parameters
    avg_time = test_rendering_speed(obj_path, num_iterations=1, image_size=512, 
                                    sigma=1e-4, gamma=1, 
                                    output_path="output/pytorch3d_output.png")
    