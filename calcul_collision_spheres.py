from urdfpy import URDF
import numpy as np
import json
import trimesh
from pathlib import Path

# numpy 임시 패치
np.int = int

def get_collision_sphere_from_mesh(mesh, scale):
    vertices = np.array(mesh.vertices) * scale
    center = np.mean(vertices, axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))
    return center.tolist(), float(radius)

def get_collision_spheres_from_urdf(urdf_path, mesh_dir=None, default_radius=0.05):
    robot = URDF.load(urdf_path)
    spheres_dict = {}

    for link in robot.links:
        link_name = link.name
        if not link.collisions:
            continue

        spheres_dict[link_name] = []

        for collision in link.collisions:
            geometry = collision.geometry
            origin = collision.origin if collision.origin is not None else np.zeros(6)
            xyz = origin[:3, 3]

            center, radius = None, None

            if geometry.mesh:
                filename = geometry.mesh.filename
                #scale = np.array(geometry.mesh.scale or [1.0, 1.0, 1.0])
                scale = np.array(geometry.mesh.scale if geometry.mesh.scale is not None else [1.0, 1.0, 1.0])


                if mesh_dir:
                    mesh_path = Path(mesh_dir) / Path(filename).name
                else:
                    mesh_path = Path(filename)

                try:
                    mesh = trimesh.load(str(mesh_path), force='mesh')
                    center, radius = get_collision_sphere_from_mesh(mesh, scale)
                except Exception as e:
                    print(f"[Warning] Failed to load mesh for {mesh_path}: {e}")
                    center, radius = [0.0, 0.0, 0.0], default_radius

            elif geometry.box:
                size = np.array(geometry.box.size)
                center = [0.0, 0.0, 0.0]
                radius = float(np.linalg.norm(size / 2.0))

            elif geometry.cylinder:
                h = geometry.cylinder.length
                r = geometry.cylinder.radius
                center = [0.0, 0.0, 0.0]
                radius = float(np.linalg.norm([r, r, h / 2.0]))

            elif geometry.sphere:
                center = [0.0, 0.0, 0.0]
                radius = geometry.sphere.radius

            else:
                continue

            # apply transform
            center = (np.array(center) + xyz).tolist()

            spheres_dict[link_name].append({
                "center": [round(c, 4) for c in center],
                "radius": round(radius, 4),
            })

    return spheres_dict


if __name__ == "__main__":
    urdf_path = "deps/OmniGibson/omnigibson/data/assets/models/kinova_gen3_lite/gen3_lite.urdf"
    mesh_dir = "deps/OmniGibson/omnigibson/data/assets/models/kinova_gen3_lite/meshes/"  # ex: "meshes/"
    collision_spheres = get_collision_spheres_from_urdf(urdf_path, mesh_dir=mesh_dir)

    # Save or print
    with open("collision_spheres.json", "w") as f:
        json.dump(collision_spheres, f, indent=2)

