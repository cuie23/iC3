import trimesh
from pathlib import Path

# python3 examples/resources/multifinger_hand/stl_to_obj.py

def stl_to_obj(input_path: str, output_path: str):
    mesh = trimesh.load_mesh(input_path)
    mesh.export(output_path, file_type='obj')


PATH_IN = "examples/resources/multifinger_hand/urdf/comfree_objects/stick.stl"
PATH_OUT = "examples/resources/multifinger_hand/urdf/comfree_objects/stick.obj"

stl_to_obj(PATH_IN, PATH_OUT)