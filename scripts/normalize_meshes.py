# -*- coding:utf-8 -*-

"""
Normalize 3D mesh files to fit within a unit cube centered at the origin.

Example Usage:
python normalize_meshes.py --input_mesh data/dragon_vrip.ply \
                            --output_mesh data/stanford_dragon.obj

Inputs and Outputs:
- input_mesh: Path to the input mesh file (e.g., .ply, .obj)
- output_mesh: Path to the output mesh file (e.g., .ply, .obj)
"""

import argparse
import numpy as np
import trimesh

def normalize_mesh(input_mesh_path, output_mesh_path):
    # Load the mesh
    mesh = trimesh.load(input_mesh_path)

    # Get mesh extents and center
    extents = mesh.extents
    center = mesh.centroid

    # Determine the scale factor to fit within unit cube
    scale_factor = 1.0 / np.max(extents)

    # Center the mesh at origin
    mesh.apply_translation(-center)
    
    # Scale to fit within unit cube
    mesh.apply_scale(scale_factor)

    # Export the normalized mesh
    mesh.export(output_mesh_path)
    print(f"Normalized mesh saved to {output_mesh_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize 3D mesh files")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument("--output_mesh", type=str, required=True, help="Path to the output mesh file")
    args = parser.parse_args()

    normalize_mesh(args.input_mesh, args.output_mesh)
