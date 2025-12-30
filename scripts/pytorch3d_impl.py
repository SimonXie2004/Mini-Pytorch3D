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
    TexturesVertex
)

def test_rendering_speed(obj_path, num_iterations=100, image_size=512):
    """
    Test the rendering speed of a 3D model using PyTorch3D
    
    Args:
        obj_path: Path to the OBJ file
        num_iterations: Number of rendering iterations for timing
        image_size: Size of the rendered image (height and width)
    
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
    
    # If the mesh doesn't have textures, create a simple vertex color texture
    if mesh.textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh.textures = textures
    
    # Setup camera
    R, T = look_at_view_transform(dist=2.7, elev=10, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Setup rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # Setup lights
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
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
    print(f"Number of iterations: {num_iterations}")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Average rendering time: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("="*60)
    
    return avg_time_ms


if __name__ == "__main__":
    obj_path = "data/utah_teapot.obj"
    
    # Test with different image sizes
    print("Testing rendering speed for Stanford Dragon model\n")
    
    # Test with 512x512 resolution
    avg_time = test_rendering_speed(obj_path, num_iterations=1, image_size=512)
    