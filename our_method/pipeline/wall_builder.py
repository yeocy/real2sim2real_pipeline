"""
Wall Builder for adding walls to OmniGibson scenes
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch as th
import open3d as o3d
import omni
from scipy.spatial.transform import Rotation as R
from PIL import Image
from pathlib import Path
from loguru import logger as log
from icecream import ic

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
import digital_cousins.utils.transform_utils as T
from digital_cousins.utils.scene_utils import compute_relative_cam_pose_from
from our_method.utils.processing_utils import (
    prepare_output_dir, unprocess_depth_linear, compute_point_cloud_from_depth,
    shrink_mask, get_aabb_vertices
)

ic.configureOutput(includeContext=True)

# Constants
DEPTH_MAX_LIMIT_DEFAULT = 20
DEFAULT_OUTPUT_SIZE = (512, 512)
PLANE_DISTANCE_THRESHOLD = 0.01
RANSAC_ITERATIONS = 1000
STATISTICAL_OUTLIER_NEIGHBORS = 20
STATISTICAL_OUTLIER_STD_RATIO = 2.0
WALL_SCALE = [8.0, 8.0, 1.0]
FLOOR_SCALE = [10.0, 10.0, 1.0]
OG_CAM_ORI_OFFSET_EULER = [np.pi, 0, 0]


# ============================================================================
# Coordinate Transform Utilities
# ============================================================================

def transform_cam_to_og_frame(points_in_cam_frame, cam_pos, cam_quat):
    """
    Transform camera frame points to OmniGibson world frame.
    
    Args:
        points_in_cam_frame (np.ndarray): Points in camera frame (N, 3) or (3,)
        cam_pos (np.ndarray): Camera position in OG frame (3,)
        cam_quat (np.ndarray): Camera orientation in OG frame (4,)

    Returns:
        np.ndarray: Points in OG frame (N, 3)
    """
    if points_in_cam_frame.ndim == 1:
        points_in_cam_frame = points_in_cam_frame.reshape(1, -1)

    if points_in_cam_frame.shape[1] != 3:
        raise ValueError("points_in_cam_frame must have shape (N, 3)")
    
    homogeneous_points = np.column_stack(
        (points_in_cam_frame, np.ones((points_in_cam_frame.shape[0], 1)))
    )

    # OmniGibson camera is rotated 180 deg wrt x-axis (-z into camera frame)
    og_cam_ori_offset = T.euler2mat(OG_CAM_ORI_OFFSET_EULER)
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.mat2quat(og_cam_ori_offset)))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))

    points_in_og_frame = (og_cam_global_tf @ og_cam_local_tf @ homogeneous_points.T)[:3, :].T
    return points_in_og_frame


def get_rotation_from_two_vectors(source_vector, target_vector):
    """
    Calculate rotation from source_vector to target_vector.
    
    Args:
        source_vector (np.ndarray): Source vector (3,)
        target_vector (np.ndarray): Target vector (3,)
        
    Returns:
        tuple: (euler_angles, quaternion)
    """
    source_normal = source_vector / np.linalg.norm(source_vector)
    target_normal = target_vector / np.linalg.norm(target_vector)
    
    if np.allclose(source_normal, target_normal):
        return np.array([1, 0, 0]), 0.0
    
    if np.allclose(source_normal, -target_normal):
        return np.array([1, 0, 0]), np.pi
    
    rotation_axis = np.cross(source_normal, target_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    cos_angle = np.dot(source_normal, target_normal)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    rotation = R.from_rotvec(rotation_axis * angle)
    return rotation.as_euler('xyz'), rotation.as_quat()


# ============================================================================
# Wall Pose and Geometry Utilities
# ============================================================================

def get_wall_pose(cam_pos, cam_quat, wall_normal, wall_point, wall_is_vertical=True):
    """
    Calculate wall pose in OmniGibson frame from camera frame wall parameters.

    Args:
        cam_pos (np.ndarray): Camera position in OG world frame (3,)
        cam_quat (np.ndarray): Camera orientation quaternion in OG frame (4,)
        wall_normal (np.ndarray): Wall normal in camera frame (3,)
        wall_point (np.ndarray): Wall point in camera frame (3,)
        wall_is_vertical (bool): Whether to snap wall to vertical orientation

    Returns:
        tuple: (position, quaternion) in OG frame
    """
    assert og.sim.is_playing()
    ic(wall_normal, wall_point, cam_pos, cam_quat)

    og_cam_ori_offset = T.euler2mat(OG_CAM_ORI_OFFSET_EULER)
    
    # Transform wall normal to OG frame
    wall_z_dir_og_frame = T.quat2mat(cam_quat) @ og_cam_ori_offset @ wall_normal

    # Transform wall point to OG frame
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.mat2quat(og_cam_ori_offset)))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    p_og_frame = (og_cam_global_tf @ og_cam_local_tf @ np.array([*wall_point, 1.0]))[:3]
    ic(og_cam_local_tf, og_cam_global_tf, p_og_frame)

    # Select wall normal pointing toward camera
    p2cam_vec = cam_pos - p_og_frame
    if np.dot(p2cam_vec, wall_z_dir_og_frame) <= 0:
        wall_z_dir_og_frame = -wall_z_dir_og_frame

    if wall_is_vertical:
        # Snap to vertical (no slant)
        wall_z_dir_og_frame[-1] = 0
        wall_z_dir_og_frame = wall_z_dir_og_frame / np.linalg.norm(wall_z_dir_og_frame)

    rot_euler, rot_quat = get_rotation_from_two_vectors(
        source_vector=np.array([0, 0, 1]),
        target_vector=wall_z_dir_og_frame
    )
    return p_og_frame, rot_quat


def make_plane_orthogonal(plane_obj):
    """
    Make plane object orthogonal to ground plane by removing z-rotation.
    
    Args:
        plane_obj: Plane object to adjust
    """
    current_position, current_orientation = plane_obj.get_position_orientation()
    ic(current_position, current_orientation)

    orientation_euler = T.mat2euler(T.quat2mat(current_orientation))
    ic(orientation_euler)

    delta_rotation_euler = np.array([0, 0, -orientation_euler[2]])
    delta_rotation_mat = T.euler2mat(delta_rotation_euler)
    ic(delta_rotation_euler, delta_rotation_mat)

    new_orientation_euler = T.mat2euler(T.quat2mat(current_orientation) @ delta_rotation_mat)
    new_orientation = T.euler2quat(new_orientation_euler)
    ic(new_orientation_euler, new_orientation)
    
    plane_obj.set_position_orientation(position=None, orientation=new_orientation)


# ============================================================================
# Texture and Material Utilities  
# ============================================================================

def set_albedo_texture_direct(obj, texture_path):
    """
    Set albedo texture directly on object's default material.
    
    Args:
        obj: Object to apply texture to
        texture_path (str): Path to texture file
    """
    og.sim.pause()
    
    for _ in range(10):
        og.sim.step()
    
    try:
        stage = og.sim.stage
        obj_path = obj.prim.GetPath()
        material_path = f"{obj_path}/Looks/default"
        
        material_prim = stage.GetPrimAtPath(material_path)
        
        if material_prim.IsValid():
            shader_prim = None
            for child in material_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader_prim = child
                    break
            
            if shader_prim:
                shader = lazy.pxr.UsdShade.Shader(shader_prim)
                tex_input = shader.GetInput('diffuse_texture')
                if tex_input:
                    tex_input.Set(texture_path)
            else:
                print("No shader prim found, creating new shader")
                create_shader_in_material(material_prim, texture_path)
        else:
            print(f"Material prim not found at: {material_path}")
            
        for _ in range(10):
            og.sim.step()
            
    except Exception as e:
        print(f"Texture 설정 오류: {e}")
        import traceback
        traceback.print_exc()
    
    og.sim.play()


def create_shader_in_material(material_prim, texture_path):
    """
    Create new shader in material prim.
    
    Args:
        material_prim: Material prim to add shader to
        texture_path (str): Path to texture file
    """
    try:
        stage = material_prim.GetStage()
        material_path = material_prim.GetPath()
        shader_path = f"{material_path}/Shader"
        
        print(f"Creating shader at: {shader_path}")
        
        shader = lazy.pxr.UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        albedo_input = shader.CreateInput('diffuse_texture', lazy.pxr.Sdf.ValueTypeNames.Asset)
        albedo_input.Set(texture_path)
        
        diffuse_input = shader.CreateInput('diffuseColor', lazy.pxr.Sdf.ValueTypeNames.Color3f)
        diffuse_input.Set((1.0, 1.0, 1.0))
        
        material = lazy.pxr.UsdShade.Material(material_prim)
        surface_output = material.CreateSurfaceOutput()
        surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")
        
        print(f"Created new shader with diffuse_texture: {shader_path}")
        
    except Exception as e:
        print(f"Shader 생성 오류: {e}")
        import traceback
        traceback.print_exc()


def get_wall_textures(input_path, step_1_output_path, wall_mask_paths, is_floor=False):
    """
    Extract wall textures from RGB image using wall masks.
    
    Args:
        input_path (str): Path to input RGB image
        step_1_output_path (str): Path to step1 output
        wall_mask_paths (list): List of wall mask paths
        is_floor (bool): Whether processing floor texture
        
    Returns:
        dict: Mapping of wall mask path to texture
    """
    wall_textures = {}
    rectifier = ImprovedWallTextureRectifier(
        input_path=input_path, 
        visualize=False, 
        debug_print=False, 
        save_intermediate=False
    )

    for wall_idx, wall_mask_path in enumerate(wall_mask_paths):
        result = rectifier.process_multiple_walls_with_mbr(
            step_1_output_path, 
            wall_index=wall_idx, 
            is_floor=is_floor
        )
        wall_textures[wall_mask_path] = result["wall_texture"]

    return wall_textures


# ============================================================================
# Point Cloud Processing
# ============================================================================

def get_wall_pcd(rgb_path, step_1_output_path, cam_pos, cam_quat, scene, 
                 depth_max_limit=DEPTH_MAX_LIMIT_DEFAULT):
    """
    Process wall point cloud from step 1 output.
    
    Args:
        rgb_path (str): Path to RGB image
        step_1_output_path (str): Path to step 1 output
        cam_pos (np.ndarray): Camera position
        cam_quat (np.ndarray): Camera quaternion
        scene: OmniGibson scene
        depth_max_limit (float): Maximum depth limit
    """
    with open(step_1_output_path, "r") as f:
        step_1_output_info = json.load(f)

    camera_intrinsics_matrix = np.array(step_1_output_info["K"])
    wall_mask_planes = step_1_output_info["wall_mask_planes"]
    raw_wall_mask_paths = list(wall_mask_planes.keys())

    floor_mask_path = step_1_output_info["floor_mask"]
    floor_mask = np.array(Image.open(floor_mask_path))
    ic(floor_mask.shape)

    depth_path = os.path.join(os.path.dirname(step_1_output_path), "step_1_depth.png")
    depth_limits = np.array([0, depth_max_limit])

    rgb = np.array(Image.open(rgb_path))
    ic(rgb.shape)
    depth = unprocess_depth_linear(np.array(Image.open(depth_path)), out_limits=depth_limits)
    pc = compute_point_cloud_from_depth(depth=depth, K=camera_intrinsics_matrix)

    pc_floor = pc.reshape(-1, 3)[floor_mask.flatten().nonzero()[0]]
    rgb_floor = rgb.reshape(-1, 3)[floor_mask.flatten().nonzero()[0]]

    pcd = o3d.geometry.PointCloud()
    pc_floor_mean = np.mean(pc_floor, axis=0)
    pcd.points = o3d.utility.Vector3dVector(pc_floor - pc_floor_mean.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgb_floor / 255.0)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=10)
    [a, b, c, d] = plane_model
    z_dir_plane = np.array([a, b, c])

    pcd_floor = o3d.geometry.PointCloud()
    pcd_floor.points = o3d.utility.Vector3dVector(pc_floor)
    pcd_floor.colors = o3d.utility.Vector3dVector(rgb_floor / 255.0)
    
    floor_bbox = get_aabb_vertices(pcd_floor.get_axis_aligned_bounding_box())
    floor_bbox = np.array(floor_bbox)
    ic(floor_bbox)

    print(f"Estimated floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    pc_floor = np.asarray(inlier_cloud.points)
    origin_pos = pc_floor[int(len(pc_floor) // 2)] + pc_floor_mean

    print(f"Selected origin_pos: {origin_pos}")

    marker_color_palatte = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]
    
    all_wall_mask_planes = dict()
    for i, wall_mask_path in enumerate(raw_wall_mask_paths):
        wall_mask = Image.open(wall_mask_path)
        shrunk_wall_mask = shrink_mask(np.array(wall_mask), iterations=2)
        pc_wall = pc.reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]]
        pcd_wall = o3d.geometry.PointCloud()
        pcd_wall.points = o3d.utility.Vector3dVector(pc_wall)
        pcd_wall.colors = o3d.utility.Vector3dVector(
            np.array(rgb).reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]] / 255.0
        )
        pcd_wall = pcd_wall.uniform_down_sample(every_k_points=5)
        pcd_wall, _ = pcd_wall.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
        pc_wall = np.asarray(pcd_wall.points)

        plane_model, inliers = pcd_wall.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        a, b, c, d = plane_model
        wall_normal_vec = np.array([a, b, c])
        wall_normal_vec = wall_normal_vec / np.linalg.norm(wall_normal_vec)
        start_point = np.median(pc_wall, axis=0)
        wall_normal_vec = -np.sign(np.dot(wall_normal_vec, start_point)) * wall_normal_vec
        all_wall_mask_planes[wall_mask_path] = {"normal": wall_normal_vec, "point": start_point}

        print(f"Estimated wall {i}'s plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        wall_bbox = get_aabb_vertices(pcd_wall.get_axis_aligned_bounding_box())
        wall_bbox = np.array(wall_bbox)

        wal_bbox_og_frame = transform_cam_to_og_frame(
            points_in_cam_frame=wall_bbox,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
        )
        ic(wal_bbox_og_frame)


# ============================================================================
# WallBuilder Class
# ============================================================================

class WallBuilder:
    """Builds walls in OmniGibson scenes using real-world extraction data."""
    
    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): Whether to print verbose output
        """
        self.verbose = verbose
        
    def __call__(self, input_path, step_1_output_path, step_3_output_path,
                 scene_info_path, save_dir, depth_max_limit=DEPTH_MAX_LIMIT_DEFAULT):
        """
        Build walls in the scene using step 1 output data.
        
        Args:
            input_path (str): Path to input RGB image
            step_1_output_path (str): Path to step 1 output containing wall information
            step_3_output_path (str): Path to step 3 output containing scene information
            scene_info_path (str): Path to scene info
            save_dir (str): Directory to save wall building results
            depth_max_limit (float): Maximum depth limit for wall point cloud processing
            
        Returns:
            tuple: (success: bool, output_path: str)
        """
        self.input_path = self._prepare_input_path(input_path)
        self.save_dir = prepare_output_dir(self.input_path, save_dir, "wall_builder_output")

        if self.verbose:
            log.debug(f"Building walls using input: {self.input_path}")
            log.debug(f"Step 1 output: {step_1_output_path}")
            log.debug(f"Step 3 output: {step_3_output_path}")
            log.debug(f"Save directory: {self.save_dir}")
            
        log.info("""
####################################################
####  Generating simulated scenes in OmniGibson ####
####################################################
        """)

        og.launch()
        scene = self.load_cousin_scene(scene_info=scene_info_path, visual_only=False)

        for obj in scene.objects:
            obj.keep_still()
        og.sim.step_physics()

        self.build_walls(scene=scene, step_1_output_path=step_1_output_path)
        
        wall_builder_info = {
            "step_1_output_path": step_1_output_path,
            "step_3_output_path": step_3_output_path,
            "depth_max_limit": depth_max_limit,
            "success": True
        }
        
        with open(os.path.join(self.save_dir, "wall_builder_info.json"), "w") as f:
            json.dump(wall_builder_info, f, indent=2)
        
        if self.verbose:
            log.debug(f"Successfully built walls in the scene")
            log.debug(f"Wall builder output saved to: {self.save_dir}")

        for _ in range(10000000):
            og.sim.render()

        return True, self.save_dir
    
    def _prepare_input_path(self, input_path):
        """Prepare input path by adding _resize suffix."""
        name_seg = input_path.split(".")
        name_seg[-2] += "_resize"
        return ".".join(name_seg)

    def create_scene(self, floor=True, sky=True):
        """
        Create new empty scene in OmniGibson.

        Args:
            floor (bool): Whether to use floor
            sky (bool): Whether to use sky

        Returns:
            Scene: OmniGibson scene
        """
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    def load_cousin_scene(self, scene_info, visual_only=False):
        """
        Load cousin scene specified by scene_info.

        Args:
            scene_info (dict or str): Scene information or path to scene info JSON
            visual_only (bool): Whether to load objects as visual only

        Returns:
            Scene: loaded OmniGibson scene
        """
        scene = self.create_scene(floor=True)

        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(f)

        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(
            th.tensor(cam_pose[0], dtype=th.float), 
            th.tensor(cam_pose[1], dtype=th.float)
        )

        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                print(f"Object Name: {obj_name}")
                print(f"Category: {obj_info['category']}")
                print(f"Model: {obj_info['model']}")
                print(f"Visual Only: {visual_only}")
                print(f"Scale: {obj_info['scale']}")

                obj = DatasetObject(
                    name=obj_name,
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=visual_only,
                    scale=obj_info["scale"]
                )
                scene.add_object(obj)
                obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                    pose_A=np.array(obj_info["tf_from_cam"]),
                    pose_A_in_B=T.pose2mat(cam_pose),
                ))
                obj.set_position_orientation(
                    th.tensor(obj_pos, dtype=th.float), 
                    th.tensor(obj_quat, dtype=th.float)
                )
        
        og.sim.step()
        return scene

    def build_walls(self, scene, step_1_output_path):
        """
        Build walls in the scene.

        Args:
            scene: OmniGibson scene to build walls in
            step_1_output_path (str): Path to step 1 output file
        """
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories = json.load(f)

        seg_dir = detected_categories["segmentation_dir"]
        K = np.array(step_1_output_info["K"])
        rgb = np.array(Image.open(step_1_output_info["input_rgb"]))
        raw_depth = np.array(Image.open(step_1_output_info["input_depth"]))
        depth_limits = np.array(step_1_output_info["depth_limits"])
        depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
        pc = compute_point_cloud_from_depth(depth=depth, K=K)

        z_dir = np.array(step_1_output_info["z_direction"])
        floor_mask_path = step_1_output_info["floor_mask"]
        wall_mask_planes = step_1_output_info["wall_mask_planes"]
        origin_pos = np.array(step_1_output_info["origin_pos"])

        cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)
        wall_mount_fpaths = list(wall_mask_planes.keys())

        # Floor texture and plane
        self._build_floor(
            floor_mask_path, step_1_output_path, 
            cam_pos, cam_quat, z_dir, origin_pos, scene
        )

        # Wall point clouds
        get_wall_pcd(
            rgb_path=self.input_path,
            step_1_output_path=step_1_output_path,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            scene=scene
        )

        # Wall textures and planes
        self._build_walls(
            wall_mount_fpaths, wall_mask_planes, 
            step_1_output_path, cam_pos, cam_quat, scene
        )

    def _build_floor(self, floor_mask_path, step_1_output_path, 
                     cam_pos, cam_quat, z_dir, origin_pos, scene):
        """Build floor plane with texture."""
        floor_textures = get_wall_textures(
            input_path=self.input_path,
            step_1_output_path=step_1_output_path,
            wall_mask_paths=[floor_mask_path],
            is_floor=True
        )
        
        floor_texture_dir = Path(self.save_dir).joinpath("wall_textures")
        floor_texture_dir.mkdir(parents=True, exist_ok=True)
        floor_texture_path = floor_texture_dir.joinpath(f"floor_texture.png")
        Image.fromarray(floor_textures[floor_mask_path]).save(floor_texture_path)
        
        p_og_frame, rot_quat = get_wall_pose(
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            wall_normal=z_dir,
            wall_point=origin_pos,
            wall_is_vertical=False,
        )
        
        og.sim.pause()
        floor = og.objects.PrimitiveObject(
            name=f"floor",
            primitive_type="Plane",
            scale=FLOOR_SCALE
        )
        scene.add_object(floor)
        floor.set_position_orientation(
            th.tensor(p_og_frame, dtype=th.float),
            th.tensor(rot_quat, dtype=th.float),
        )
        set_albedo_texture_direct(floor, texture_path=str(floor_texture_path))

        # Hide ground plane
        stage = omni.usd.get_context().get_stage()
        ground_plane_prim = stage.GetPrimAtPath("/World/ground_plane")
        if ground_plane_prim:
            imageable = lazy.pxr.UsdGeom.Imageable(ground_plane_prim)
            imageable.MakeInvisible()
        og.sim.play()

    def _build_walls(self, wall_mount_fpaths, wall_mask_planes, 
                     step_1_output_path, cam_pos, cam_quat, scene):
        """Build wall planes with textures."""
        wall_textures = get_wall_textures(
            input_path=self.input_path,
            step_1_output_path=step_1_output_path,
            wall_mask_paths=wall_mount_fpaths
        )
        
        wall_texture_dir = Path(self.save_dir).joinpath("wall_textures")
        wall_texture_dir.mkdir(parents=True, exist_ok=True)
        wall_texture_paths_dict = {}
        
        for wall_i, (wall_mount_fpath, wall_texture) in enumerate(wall_textures.items()):
            wall_texture_path = wall_texture_dir.joinpath(f"wall_texture_{wall_i}.png")
            Image.fromarray(wall_texture).save(wall_texture_path)
            wall_texture_paths_dict[wall_mount_fpath] = str(wall_texture_path)
        
        with open(wall_texture_dir.joinpath("wall_textures_dictionary.json"), "w") as f:
            json.dump(wall_texture_paths_dict, f, indent=4)

        for wall_i, wall_mount_fpath in enumerate(wall_mount_fpaths):
            p_og_frame, rot_quat = get_wall_pose(
                cam_pos=cam_pos,
                cam_quat=cam_quat,
                wall_normal=wall_mask_planes[wall_mount_fpath]["normal"],
                wall_point=wall_mask_planes[wall_mount_fpath]["point"],
                wall_is_vertical=True,
            )

            og.sim.pause()
            wall = og.objects.PrimitiveObject(
                name=f"wall{wall_i}",
                primitive_type="Plane",
                rgba=[1.0, 1.0, 1.0, 1.0],
                scale=WALL_SCALE
            )
            scene.add_object(wall)
            wall.set_position_orientation(
                th.tensor(p_og_frame, dtype=th.float),
                th.tensor(rot_quat, dtype=th.float),
            )
            make_plane_orthogonal(wall)
            set_albedo_texture_direct(wall, texture_path=wall_texture_paths_dict[wall_mount_fpath])
            og.sim.play()


# ============================================================================
# ImprovedWallTextureRectifier Class
# ============================================================================

class ImprovedWallTextureRectifier:
    """Rectify and extract wall textures from perspective images."""
    
    def __init__(self, input_path=None, visualize=True, debug_print=True, 
                 save_intermediate=False):
        """
        Args:
            input_path (str): Input image path
            visualize (bool): Whether to show visualizations
            debug_print (bool): Whether to print debug messages
            save_intermediate (bool): Whether to save intermediate results
        """
        self.input_path = input_path
        self.homography_matrix = None
        self.visualize = visualize
        self.debug_print = debug_print
        self.save_intermediate = save_intermediate
        
    def _print_debug(self, message):
        """Print debug message if enabled."""
        if self.debug_print:
            print(message)
            
    def _save_debug_image(self, image, filename):
        """Save debug image if enabled."""
        if self.save_intermediate:
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(filename)
            else:
                image.save(filename)
            self._print_debug(f"Saved debug image: {filename}")
        
    def load_data_from_paths(self, step1_output_path, is_floor=False):
        """
        Load data from step 1 output paths.
        
        Args:
            step1_output_path (str): Path to step 1 output JSON
            is_floor (bool): Whether processing floor plane
            
        Returns:
            tuple: (image, point_cloud, masks, output_info, floor_normal)
        """
        image_path = self.input_path
        image = np.array(Image.open(image_path))
        
        depth_image_path = os.path.join(os.path.dirname(step1_output_path), "step_1_depth.png")
        step1_output_info_path = step1_output_path
        
        depth_image = Image.open(depth_image_path)
        depth_array = np.array(depth_image)
        
        self._print_debug(f"depth array shape: {depth_array.shape}")
        
        with open(step1_output_info_path, 'r') as f:
            step1_output_info = json.load(f)
        
        camera_intrinsic = step1_output_info["K"]
        self._print_debug(f"camera intrinsic: {camera_intrinsic}")
        
        depth_image_o3d = o3d.geometry.Image(depth_array.astype(np.float32))
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=depth_array.shape[1], 
            height=depth_array.shape[0], 
            fx=camera_intrinsic[0][0], 
            fy=camera_intrinsic[1][1], 
            cx=camera_intrinsic[0][2], 
            cy=camera_intrinsic[1][2]
        )
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, intrinsic)
        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points.reshape((depth_array.shape[0], depth_array.shape[1], 3))
        
        floor_mask_path = step1_output_info["floor_mask"]
        floor_mask = np.array(Image.open(floor_mask_path))
        z_direction = step1_output_info["z_direction"]

        wall_mask_plane = step1_output_info["wall_mask_planes"]
        wall_mask_paths = list(wall_mask_plane.keys())
        wall_masks = [np.array(Image.open(wall_mask_path)) for wall_mask_path in wall_mask_paths]
        
        if is_floor:
            return image, pcd_points, [floor_mask], step1_output_info, [-1, 0, 0]
        else:
            return image, pcd_points, wall_masks, step1_output_info, z_direction
        
    def extract_wall_region(self, image, segmentation_mask):
        """Extract wall region using segmentation mask."""
        wall_mask = (segmentation_mask > 0).astype(np.uint8)
        wall_region = cv2.bitwise_and(image, image, mask=wall_mask)
        
        if self.visualize:
            self._visualize_wall_extraction(image, wall_mask, wall_region)
        
        return wall_region, wall_mask
    
    def _visualize_wall_extraction(self, image, wall_mask, wall_region):
        """Visualize wall region extraction process."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(wall_mask, cmap='gray')
        axes[1].set_title('Wall Segmentation Mask')
        axes[1].axis('off')
        
        axes[2].imshow(wall_region)
        axes[2].set_title('Extracted Wall Region')
        axes[2].axis('off')
        
        plt.suptitle('Step 1: Wall Region Extraction')
        plt.tight_layout()
        plt.show()
    
    def estimate_wall_normal_with_o3d(self, points, method='pca'):
        """
        Estimate wall normal vector using Open3D.
        
        Args:
            points (np.ndarray): 3D points (N x 3)
            method (str): 'pca', 'ransac', or 'both'
            
        Returns:
            tuple: (normal_vector, cleaned_points)
        """
        if len(points) < 3:
            raise ValueError("At least 3 points required for normal estimation")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=STATISTICAL_OUTLIER_NEIGHBORS, 
            std_ratio=STATISTICAL_OUTLIER_STD_RATIO
        )
        clean_points = np.asarray(pcd.points)
        
        if method == 'pca' or method == 'both':
            centroid = np.mean(clean_points, axis=0)
            centered_points = clean_points - centroid
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal_pca = eigenvectors[:, 0]
            planarity = 1 - eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 1e-10 else 0
            
            if self.debug_print:
                self._print_debug(f"PCA Planarity score: {planarity:.3f}")
                self._print_debug(f"Eigenvalues: {eigenvalues}")
        
        if method == 'ransac' or method == 'both':
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=PLANE_DISTANCE_THRESHOLD,
                ransac_n=3,
                num_iterations=RANSAC_ITERATIONS
            )
            normal_ransac = np.array(plane_model[:3])
            inlier_ratio = len(inliers) / len(clean_points)
            
            if self.debug_print:
                self._print_debug(f"RANSAC inlier ratio: {inlier_ratio:.3f}")
                self._print_debug(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
        
        if method == 'pca':
            final_normal = normal_pca
        elif method == 'ransac':
            final_normal = normal_ransac
        else:
            final_normal = normal_ransac if inlier_ratio > 0.7 else normal_pca
        
        camera_direction = np.array([0, 0, -1])
        if np.dot(final_normal, camera_direction) < 0:
            final_normal = -final_normal
        
        if self.visualize:
            self._visualize_normal_estimation_o3d(pcd, final_normal, clean_points)
        
        return final_normal, clean_points
        
        if method == 'ransac' or method == 'both':
            # RANSAC을 사용한 더 강건한 평면 추정
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01,  # 평면으로부터의 거리 임계값
                ransac_n=3,               # RANSAC을 위한 최소 점 개수
                num_iterations=1000       # RANSAC 반복 횟수
            )
            
            # plane_model = [a, b, c, d] where ax + by + cz + d = 0
            normal_ransac = np.array(plane_model[:3])
            inlier_ratio = len(inliers) / len(clean_points)
            
            if self.debug_print:
                self._print_debug(f"RANSAC inlier ratio: {inlier_ratio:.3f}")
                self._print_debug(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
        
        # 최종 normal vector 선택
        if method == 'pca':
            final_normal = normal_pca
        elif method == 'ransac':
            final_normal = normal_ransac
        else:  # both
            # RANSAC 결과가 좋으면 RANSAC 사용, 아니면 PCA 사용
            if inlier_ratio > 0.7:
                final_normal = normal_ransac
            else:
                final_normal = normal_pca
        
        # Normal vector 방향 조정 (카메라를 향하도록)
        camera_direction = np.array([0, 0, -1])
        if np.dot(final_normal, camera_direction) < 0:
            final_normal = -final_normal
        
        if self.visualize:
            self._visualize_normal_estimation_o3d(pcd, final_normal, clean_points)
        
        return final_normal, clean_points
    
    def _visualize_normal_estimation_o3d(self, pcd, normal_vector, points):
        """Visualize normal vector estimation using Open3D."""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D 점들과 normal vector
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.6, s=1, c='blue')
        
        centroid = np.mean(points, axis=0)
        ax1.scatter(*centroid, color='red', s=100, label='Centroid')
        
        # Normal vector 표시
        scale = 0.5
        ax1.quiver(centroid[0], centroid[1], centroid[2], 
                  normal_vector[0]*scale, normal_vector[1]*scale, normal_vector[2]*scale,
                  color='red', arrow_length_ratio=0.1, linewidth=3, label='Normal Vector')
        
        ax1.set_title('3D Points and Estimated Normal (Open3D)')
        ax1.legend()
        
        # 점들의 분포 히스토그램
        ax2 = fig.add_subplot(132)
        distances_to_plane = np.abs(np.dot(points - centroid, normal_vector))
        ax2.hist(distances_to_plane, bins=50, alpha=0.7)
        ax2.set_xlabel('Distance to Plane')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Distance Distribution')
        ax2.axvline(np.mean(distances_to_plane), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances_to_plane):.4f}')
        ax2.legend()
        
        # Normal vector 성분
        ax3 = fig.add_subplot(133)
        components = ['X', 'Y', 'Z']
        ax3.bar(components, normal_vector, color=['red', 'green', 'blue'])
        ax3.set_title('Normal Vector Components')
        ax3.set_ylabel('Component Value')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Open3D-based Normal Vector Estimation')
        plt.tight_layout()
        plt.show()
    
    def estimate_wall_plane_with_o3d(self, point_cloud, normal_vector, mask):
        """Estimate wall plane efficiently using Open3D."""
        valid_points = point_cloud[mask > 0]
        
        if normal_vector is None:
            wall_normal, clean_points = self.estimate_wall_normal_with_o3d(valid_points, method='both')
            self._print_debug(f"Estimated wall normal vector: {wall_normal}")
        else:
            wall_normal = normal_vector / np.linalg.norm(normal_vector)
            clean_points = valid_points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clean_points)
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=STATISTICAL_OUTLIER_NEIGHBORS, 
            std_ratio=STATISTICAL_OUTLIER_STD_RATIO
        )
        
        wall_center = np.mean(np.asarray(pcd_clean.points), axis=0)
        
        if self.visualize:
            self._visualize_wall_plane_o3d(pcd_clean, wall_normal, wall_center)
        
        return wall_normal, wall_center, np.asarray(pcd_clean.points)
    
    def _visualize_wall_plane_o3d(self, pcd_clean, wall_normal, wall_center):
        """Visualize wall plane using Open3D results."""
        points = np.asarray(pcd_clean.points)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 정리된 벽면 포인트들
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  alpha=0.6, s=1, c='blue', label='Cleaned Wall Points')
        
        # 벽면 중심과 normal vector
        ax.scatter(*wall_center, color='red', s=100, label='Wall Center')
        
        scale = 0.5
        ax.quiver(wall_center[0], wall_center[1], wall_center[2],
                 wall_normal[0]*scale, wall_normal[1]*scale, wall_normal[2]*scale,
                 color='red', arrow_length_ratio=0.1, linewidth=3, label='Wall Normal')
        
        distances_to_plane = np.abs(np.dot(points - wall_center, wall_normal))
        rmse = np.sqrt(np.mean(distances_to_plane**2))
        
        ax.set_title(f'Wall Plane (Open3D) - RMSE: {rmse:.4f}m')
        ax.legend()
        plt.show()
        
        self._print_debug(f"Plane fit RMSE: {rmse:.4f}m")
        self._print_debug(f"Max distance: {np.max(distances_to_plane):.4f}m")
        self._print_debug(f"95% of points within {np.percentile(distances_to_plane, 95):.4f}m")
    
    def find_minimum_bounding_rectangle_3d(self, wall_points, wall_normal, wall_center, floor_normal=None):
        """
        Find Minimum Bounding Rectangle in 3D and return corner points.
        Uses floor normal for accurate vertical direction calculation.
        
        Args:
            wall_points (np.ndarray): Wall 3D points (N x 3)
            wall_normal (np.ndarray): Wall normal vector (3,)
            wall_center (np.ndarray): Wall center point (3,)
            floor_normal (np.ndarray): Floor normal vector (3,) - None for default
            
        Returns:
            np.ndarray: Rectangle 4 corner points (4 x 3)
        """
        self._print_debug("=== Finding Minimum Bounding Rectangle in 3D (with Floor Normal) ===")
        
        if floor_normal is not None:
            self._print_debug(f"Using floor normal: {floor_normal}")
            
            # Project floor normal to wall plane to get vertical direction
            vertical_axis = floor_normal - np.dot(floor_normal, wall_normal) * wall_normal
            
            if np.linalg.norm(vertical_axis) < 1e-6:
                self._print_debug("Warning: Floor normal nearly parallel to wall. Using gravity assumption.")
                gravity_vector = np.array([0, -1, 0])
                vertical_axis = -gravity_vector + np.dot(gravity_vector, wall_normal) * wall_normal
            
            vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
            
            # Ensure vertical axis points upward
            if np.dot(vertical_axis, floor_normal) < 0:
                vertical_axis = -vertical_axis
                
        else:
            self._print_debug("Floor normal not provided. Using camera coordinate system.")
            up_vector = np.array([0, -1, 0])
            vertical_axis = up_vector - np.dot(up_vector, wall_normal) * wall_normal
            vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
        
        horizontal_axis = np.cross(wall_normal, vertical_axis)
        horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
        
        if self.debug_print:
            self._print_debug(f"Horizontal axis: {horizontal_axis}")
            self._print_debug(f"Vertical axis: {vertical_axis}")
            self._print_debug(f"Normal axis: {wall_normal}")
            dot_h_v = np.dot(horizontal_axis, vertical_axis)
            dot_h_n = np.dot(horizontal_axis, wall_normal)
            dot_v_n = np.dot(vertical_axis, wall_normal)
            self._print_debug(f"Orthogonality: H·V={dot_h_v:.6f}, H·N={dot_h_n:.6f}, V·N={dot_v_n:.6f}")
        
        # Transform 3D points to 2D coordinates
        relative_points = wall_points - wall_center
        u_coords = np.dot(relative_points, horizontal_axis)
        v_coords = np.dot(relative_points, vertical_axis)
        points_2d = np.column_stack([u_coords, v_coords]).astype(np.float32)
        
        self._print_debug(f"2D coordinates range: u=[{np.min(u_coords):.3f}, {np.max(u_coords):.3f}], v=[{np.min(v_coords):.3f}, {np.max(v_coords):.3f}]")
        
        # 3. OpenCV의 minAreaRect로 Minimum Bounding Rectangle 찾기
        rect = cv2.minAreaRect(points_2d)
        box_2d = cv2.boxPoints(rect)
        
        center_2d, size, angle = rect
        if self.debug_print:
            self._print_debug(f"MBR center (2D): {center_2d}")
            self._print_debug(f"MBR size: {size}")
            self._print_debug(f"MBR rotation angle: {angle:.1f} degrees")
        
        # 4. 2D 직사각형 모서리점들을 3D로 변환
        corners_3d = []
        for i, point_2d in enumerate(box_2d):
            # 2D 평면 좌표를 3D로 변환
            point_3d = (wall_center + 
                    point_2d[0] * horizontal_axis + 
                    point_2d[1] * vertical_axis)
            corners_3d.append(point_3d)
        
        corners_3d = np.array(corners_3d)
        
        # 5. 모서리점들을 시계방향으로 정렬
        corners_3d = self._sort_corners_clockwise(corners_3d, wall_center, horizontal_axis, vertical_axis)
        
        if self.visualize:
            self._visualize_mbr_3d_with_floor(wall_points, corners_3d, wall_center, wall_normal, 
                                            horizontal_axis, vertical_axis, points_2d, box_2d, floor_normal)
        
        return corners_3d

    def find_minimum_bounding_rectangle_3d_floor(self, wall_points, wall_normal, wall_center):
        self._print_debug("=== Finding Minimum Bounding Rectangle in 3D for floor plane ===")
        wall_normal = -wall_normal  # floor plane에서 normal은 down 방향
        
        # 1. 벽면 평면의 로컬 좌표계 생성
        # 기존 방식: 카메라 좌표계 기준 위쪽 벡터 사용
        up_vector = np.array([-1, 0, 0])  # 카메라 좌표계에서 왼쪽
        
        # up_vector를 벽면에 투영하여 수직축 생성
        vertical_axis = up_vector - np.dot(up_vector, wall_normal) * wall_normal
        vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
        
        # 수평축은 normal과 수직축의 외적
        horizontal_axis = np.cross(wall_normal, vertical_axis)
        horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
        
        if self.debug_print:
            self._print_debug(f"Horizontal axis: {horizontal_axis}")
            self._print_debug(f"Vertical axis: {vertical_axis}")
            self._print_debug(f"Normal axis: {wall_normal}")
            
            # 좌표축 직교성 검증
            dot_h_v = np.dot(horizontal_axis, vertical_axis)
            dot_h_n = np.dot(horizontal_axis, wall_normal)
            dot_v_n = np.dot(vertical_axis, wall_normal)
            self._print_debug(f"Orthogonality check: H·V={dot_h_v:.6f}, H·N={dot_h_n:.6f}, V·N={dot_v_n:.6f}")
        
        # 2. 3D 점들을 벽면 평면의 2D 좌표로 변환
        relative_points = wall_points - wall_center
        u_coords = np.dot(relative_points, horizontal_axis)
        v_coords = np.dot(relative_points, vertical_axis)
        points_2d = np.column_stack([u_coords, v_coords]).astype(np.float32)
        
        self._print_debug(f"2D coordinates range: u=[{np.min(u_coords):.3f}, {np.max(u_coords):.3f}], v=[{np.min(v_coords):.3f}, {np.max(v_coords):.3f}]")
        
        # 3. OpenCV의 minAreaRect로 Minimum Bounding Rectangle 찾기
        rect = cv2.minAreaRect(points_2d)
        box_2d = cv2.boxPoints(rect)
        
        center_2d, size, angle = rect
        if self.debug_print:
            self._print_debug(f"MBR center (2D): {center_2d}")
            self._print_debug(f"MBR size: {size}")
            self._print_debug(f"MBR rotation angle: {angle:.1f} degrees")
        
        # 4. 2D 직사각형 모서리점들을 3D로 변환
        corners_3d = []
        for i, point_2d in enumerate(box_2d):
            # 2D 평면 좌표를 3D로 변환
            point_3d = (wall_center + 
                    point_2d[0] * horizontal_axis + 
                    point_2d[1] * vertical_axis)
            corners_3d.append(point_3d)
        
        corners_3d = np.array(corners_3d)
        
        # 5. 모서리점들을 시계방향으로 정렬
        corners_3d = self._sort_corners_clockwise(corners_3d, wall_center, horizontal_axis, vertical_axis)
        
        if self.visualize:
            self._visualize_mbr_3d_with_floor(wall_points, corners_3d, wall_center, wall_normal, 
                                            horizontal_axis, vertical_axis, points_2d, box_2d, [-1, 0, 0])
        
        return corners_3d

    def _visualize_mbr_3d_with_floor(self, wall_points, corners_3d, wall_center, wall_normal, 
                                    horizontal_axis, vertical_axis, points_2d, box_2d, floor_normal):
        """
        Floor normal을 포함한 3D Minimum Bounding Rectangle 시각화
        """
        fig = plt.figure(figsize=(20, 6))
        
        # 통계 정보 (floor normal 관련 정보 추가)
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        # MBR 크기 계산
        mbr_width = np.linalg.norm(corners_3d[1] - corners_3d[0])
        mbr_height = np.linalg.norm(corners_3d[3] - corners_3d[0])
        mbr_area = mbr_width * mbr_height
        
        # Floor normal과 vertical axis 사이의 각도
        if floor_normal is not None:
            angle_with_floor = np.arccos(np.clip(np.dot(vertical_axis, floor_normal), -1, 1)) * 180 / np.pi
            floor_info = f"Floor-Vertical angle: {angle_with_floor:.1f}°"
        else:
            floor_info = "Floor normal: Not provided"
        
        # 통계 텍스트
        stats_text = f"""
    MBR Statistics (with Floor Normal):
    ──────────────────────────────────
    Width: {mbr_width:.3f} m
    Height: {mbr_height:.3f} m
    Area: {mbr_area:.3f} m²

    Points: {len(wall_points):,}
    Coverage: {len(wall_points)/1000:.1f}K points

    Coordinate System:
    ──────────────────────────────────
    Horizontal: [{horizontal_axis[0]:.2f}, {horizontal_axis[1]:.2f}, {horizontal_axis[2]:.2f}]
    Vertical: [{vertical_axis[0]:.2f}, {vertical_axis[1]:.2f}, {vertical_axis[2]:.2f}]
    Wall Normal: [{wall_normal[0]:.2f}, {wall_normal[1]:.2f}, {wall_normal[2]:.2f}]
    {f"Floor Normal: [{floor_normal[0]:.2f}, {floor_normal[1]:.2f}, {floor_normal[2]:.2f}]" if floor_normal is not None else "Floor Normal: Not provided"}

    Alignment Quality:
    ──────────────────────────────────
    {floor_info}
        """
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle('3D MBR Analysis with Floor Normal Information', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        if self.debug_print:
            self._print_debug(f"MBR 크기: {mbr_width:.3f}m x {mbr_height:.3f}m")
            self._print_debug(f"MBR 면적: {mbr_area:.3f}m²")
            if floor_normal is not None:
                self._print_debug(f"Floor-Vertical alignment: {angle_with_floor:.1f}° (0°가 완벽한 정렬)")
            
    def _sort_corners_clockwise(self, corners_3d, wall_center, horizontal_axis, vertical_axis):
        """
        3D 모서리점들을 시계방향으로 정렬 (이미지 투영시 일관된 순서를 위해)
        좌상단 → 우상단 → 우하단 → 좌하단 순서로 정렬
        """
        # 3D 점들을 2D 로컬 좌표로 변환
        relative_points = corners_3d - wall_center
        u_coords = np.dot(relative_points, horizontal_axis)
        v_coords = np.dot(relative_points, vertical_axis)
        
        # 각 점을 4개 사분면으로 분류하여 명확한 순서 보장
        # u_coords: 좌(-) → 우(+), v_coords: 상(+) → 하(-)
        
        corners_with_coords = []
        for i, (u, v) in enumerate(zip(u_coords, v_coords)):
            corners_with_coords.append((corners_3d[i], u, v, i))
        
        # 좌상단 → 우상단 → 우하단 → 좌하단 순서로 정렬
        def get_corner_order(corner_info):
            corner, u, v, idx = corner_info
            # v (수직) 기준으로 상/하 구분, u (수평) 기준으로 좌/우 구분
            if v > 0:  # 상단
                if u < 0:  # 좌상단
                    return 0
                else:      # 우상단
                    return 1
            else:      # 하단
                if u > 0:  # 우하단
                    return 2
                else:      # 좌하단
                    return 3
        
        # 순서대로 정렬
        corners_with_coords.sort(key=get_corner_order)
        sorted_corners = np.array([corner_info[0] for corner_info in corners_with_coords])
        
        return sorted_corners
    
    def project_3d_to_2d(self, points_3d, camera_intrinsic):
        """
        3D 점들을 카메라 내부 파라미터를 사용하여 2D 이미지 평면으로 투영
        
        Args:
            points_3d: 3D 점들 (N x 3)
            camera_intrinsic: 카메라 내부 파라미터 행렬 (3 x 3)
            
        Returns:
            points_2d: 투영된 2D 점들 (N x 2)
        """
        # 3D 점들을 homogeneous coordinates로 변환
        points_3d_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
        
        # 카메라 내부 파라미터 적용
        K = np.array(camera_intrinsic)
        
        # 투영: P = K * [R|t] * X, 여기서 [R|t]는 identity (카메라 좌표계에서)
        projected_homo = K @ points_3d.T  # (3 x N)
        
        # Z로 나누어 정규화 (perspective division)
        points_2d = projected_homo[:2] / projected_homo[2]  # (2 x N)
        points_2d = points_2d.T  # (N x 2)
        
        return points_2d
    
    def create_rectified_target_rectangle(self, corners_3d, output_size=(512, 512)):
        """
        3D MBR의 실제 크기 비율을 유지하여 정렬된 타겟 직사각형 생성
        corners_3d와 동일한 순서(좌상단→우상단→우하단→좌하단)로 생성
        
        Args:
            corners_3d: 3D 모서리점들 (4 x 3) - 이미 정렬된 순서
            output_size: 출력 이미지 크기 (width, height)
            
        Returns:
            target_corners: 정렬된 타겟 직사각형 모서리점들 (4 x 2)
        """
        # 3D에서의 실제 크기 계산 (정렬된 순서 기준)
        width_3d = np.linalg.norm(corners_3d[1] - corners_3d[0])   # 좌상단 → 우상단
        height_3d = np.linalg.norm(corners_3d[3] - corners_3d[0])  # 좌상단 → 좌하단
        aspect_ratio = width_3d / height_3d
        
        if self.debug_print:
            self._print_debug(f"실제 벽면 크기: {width_3d:.3f}m x {height_3d:.3f}m")
            self._print_debug(f"Aspect ratio: {aspect_ratio:.3f}")
        
        # 출력 크기에 맞춰 비율 조정
        if aspect_ratio > 1:  # 가로가 더 긴 경우
            target_width = output_size[0]
            target_height = int(output_size[0] / aspect_ratio)
        else:  # 세로가 더 긴 경우
            target_width = int(output_size[1] * aspect_ratio)
            target_height = output_size[1]
        
        # 중앙 정렬을 위한 오프셋
        offset_x = (output_size[0] - target_width) // 2
        offset_y = (output_size[1] - target_height) // 2
        
        # corners_3d와 동일한 순서로 타겟 모서리점들 생성
        # 좌상단 → 우상단 → 우하단 → 좌하단
        target_corners = np.array([
            [offset_x, offset_y],                                    # 좌상단 (corner 0)
            [offset_x + target_width, offset_y],                     # 우상단 (corner 1)
            [offset_x + target_width, offset_y + target_height],     # 우하단 (corner 2)
            [offset_x, offset_y + target_height]                     # 좌하단 (corner 3)
        ], dtype=np.float32)
        
        if self.debug_print:
            self._print_debug(f"타겟 직사각형 크기: {target_width} x {target_height}")
            self._print_debug(f"타겟 모서리점들 (좌상→우상→우하→좌하):\n{target_corners}")
        
        return target_corners
    
    def compute_homography(self, source_points, target_points):
        """
        소스 점들과 타겟 점들 사이의 homography 계산
        """
        if len(source_points) < 4 or len(target_points) < 4:
            raise ValueError("Homography 계산을 위해서는 최소 4개의 점이 필요합니다.")
        
        H, mask = cv2.findHomography(
            source_points.astype(np.float32), 
            target_points.astype(np.float32), 
            cv2.RANSAC,
            5.0
        )
        
        return H
    
    def find_largest_rectangle(self, img_array, black_threshold=0):
        """
        Find the largest rectangle in a binary image using a histogram-based approach.

        Args:
            img_array (np.ndarray): image array.
            black_threshold (int): Threshold to determine if a pixel is considered black.

        Returns:
            tuple: (max_area, (left_x, top_y, right_x, bottom_y)) of the largest rectangle.
        """
        binary = (img_array > black_threshold).astype(int)
        
        height, width = binary.shape
        max_area = 0
        max_rect = (0, 0, 0, 0)
        
        # Apply histogram-based approach to find the largest rectangle
        histogram = [0] * width
        
        for row in range(height):
            for col in range(width):
                if binary[row, col] == 1:
                    histogram[col] += 1
                else:
                    histogram[col] = 0
            
            # largest rectangle in histogram
            stack = []
            for i, h in enumerate(histogram + [0]):
                while stack and histogram[stack[-1]] > h:
                    height_idx = stack.pop()
                    rect_height = histogram[height_idx]
                    rect_width = i if not stack else i - stack[-1] - 1
                    left_x = 0 if not stack else stack[-1] + 1
                    area = rect_height * rect_width
                    
                    if area > max_area:
                        max_area = area
                        max_rect = (left_x, row - rect_height + 1, 
                                  left_x + rect_width - 1, row)
                
                stack.append(i)
        
        return max_area, max_rect
    
    def extract_wall_texture_from_rectified(self, rectified_image, rectified_mask, padding=10):
        """
        rectified된 이미지에서 순수한 벽 텍스처만 추출
        
        Args:
            rectified_image: rectified된 RGB 이미지
            rectified_mask: rectified된 세그멘테이션 마스크
            padding: 추출할 때 여백 픽셀 수
            
        Returns:
            wall_texture: 순수한 벽 텍스처 이미지
            texture_bbox: 텍스처 영역의 bounding box (x, y, w, h)
        """
        self._print_debug("=== Extracting Pure Wall Texture from Rectified Image ===")
        max_area, max_rect = self.find_largest_rectangle(img_array=rectified_mask, black_threshold=0)

        wall_texture = rectified_image[max_rect[1]:max_rect[3]+1, max_rect[0]:max_rect[2]+1]
        
        if self.save_intermediate:
            self._save_debug_image(rectified_mask, "rectified_mask.png")
            self._save_debug_image(rectified_image, "rectified_image.png")
            
        if self.visualize:
            self._visualize_texture_extraction(rectified_image, rectified_mask, max_rect)
        
        return wall_texture, max_rect
    
    def _visualize_texture_extraction(self, rectified_image, rectified_mask, rect_coords):
        """
        텍스처 추출 과정 시각화
        """
        import matplotlib.patches as patches

        fig, axs = plt.subplots(1, 3, figsize=(12, 8))

        axs[0].imshow(rectified_image)
        axs[0].set_title('Rectified Image')
        axs[0].axis('off')

        axs[1].imshow(rectified_mask, cmap='gray')
        axs[1].set_title('Rectified Mask')
        axs[1].axis('off')

        # 직사각형 영역 표시
        axs[2].imshow(rectified_image)
        axs[2].set_title('Texture Area Highlighted')
        axs[2].axis('off')
        # 직사각형 그리기
        x1, y1, x2, y2 = rect_coords
        rect = patches.Rectangle((x1, y1), x2-x1+1, y2-y1+1, 
                            linewidth=2, edgecolor='red', facecolor='none')
        axs[2].add_patch(rect)
        
        axs[2].set_title(f'biggest rectangle (area: {(x2-x1+1)*(y2-y1+1)} pixels)')
        plt.show()

    def rectify_texture_with_mbr(self, image, segmentation_mask, point_cloud, camera_intrinsic, 
                            normal_vector=None, floor_normal=None, output_size=(512, 512)):
        """
        MBR을 활용한 개선된 메인 텍스처 복원 함수 (floor normal 지원)
        """
        self._print_debug("=== Starting MBR-based Wall Texture Rectification (with Floor Normal) ===")
        
        # 1. 벽 영역 추출
        self._print_debug("Step 1: Extracting wall region...")
        wall_region, wall_mask = self.extract_wall_region(image, segmentation_mask)
        
        # 2. Open3D를 사용한 벽 평면 추정
        self._print_debug("Step 2: Estimating wall plane with Open3D...")
        wall_normal, wall_center, valid_points = self.estimate_wall_plane_with_o3d(
            point_cloud, normal_vector, wall_mask
        )
        
        # 3. 3D에서 Minimum Bounding Rectangle 찾기 (floor normal 사용)
        self._print_debug("Step 3: Finding Minimum Bounding Rectangle in 3D (with floor normal)...")
        corners_3d = self.find_minimum_bounding_rectangle_3d(
            valid_points, wall_normal, wall_center, floor_normal
        )
        
        # 4. 3D 모서리점들을 2D 이미지 평면으로 투영
        self._print_debug("Step 4: Projecting 3D corners to 2D image plane...")
        corners_2d = self.project_3d_to_2d(corners_3d, camera_intrinsic)
        
        # 5. 정렬된 타겟 직사각형 생성
        self._print_debug("Step 5: Creating aligned target rectangle...")
        target_corners = self.create_rectified_target_rectangle(corners_3d, output_size)
        
        # 6. Homography 계산
        self._print_debug("Step 6: Computing homography...")
        self.homography_matrix = self.compute_homography(corners_2d, target_corners)
        
        # 7. RGB 이미지와 마스크 모두 원근 변환 적용
        self._print_debug("Step 7: Applying perspective transformation to both image and mask...")
        rectified_image = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 마스크도 동일하게 변환
        rectified_mask = cv2.warpPerspective(
            segmentation_mask.astype(np.uint8), 
            self.homography_matrix, 
            output_size,
            flags=cv2.INTER_NEAREST,  # 마스크는 nearest neighbor 사용
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 8. 순수한 벽 텍스처 추출
        self._print_debug("Step 8: Extracting pure wall texture...")
        wall_texture, texture_bbox = self.extract_wall_texture_from_rectified(
            rectified_image, rectified_mask
        )
        
        if self.save_intermediate:
            self._save_debug_image(wall_texture, "wall_texture.png")
            
        if self.visualize:
            self._visualize_mbr_rectification_result(
                image, rectified_image, corners_3d, corners_2d, target_corners, 
                wall_mask, wall_normal, camera_intrinsic
            )
        
        return (rectified_image, rectified_mask, wall_texture, texture_bbox, 
                self.homography_matrix, corners_3d, corners_2d, target_corners, wall_normal)

    def rectify_texture_with_mbr_floor(self, image, segmentation_mask, point_cloud, camera_intrinsic, 
                            normal_vector=None, floor_normal=None, output_size=(512, 512)):
        """
        MBR을 활용한 개선된 메인 텍스처 복원 함수 (floor normal 지원)
        """
        self._print_debug("=== Starting MBR-based Wall Texture Rectification (with Floor Normal) ===")
        
        # 1. 벽 영역 추출
        self._print_debug("Step 1: Extracting wall region...")
        wall_region, wall_mask = self.extract_wall_region(image, segmentation_mask)
        
        # 2. Open3D를 사용한 벽 평면 추정
        self._print_debug("Step 2: Estimating wall plane with Open3D...")
        wall_normal, wall_center, valid_points = self.estimate_wall_plane_with_o3d(
            point_cloud, normal_vector, wall_mask
        )
        
        # 3. 3D에서 Minimum Bounding Rectangle 찾기 (floor normal 사용)
        self._print_debug("Step 3: Finding Minimum Bounding Rectangle in 3D (with floor normal)...")
        corners_3d = self.find_minimum_bounding_rectangle_3d_floor(
            valid_points, wall_normal, wall_center
        )
        
        # 4. 3D 모서리점들을 2D 이미지 평면으로 투영
        self._print_debug("Step 4: Projecting 3D corners to 2D image plane...")
        corners_2d = self.project_3d_to_2d(corners_3d, camera_intrinsic)
        
        # 5. 정렬된 타겟 직사각형 생성
        self._print_debug("Step 5: Creating aligned target rectangle...")
        target_corners = self.create_rectified_target_rectangle(corners_3d, output_size)
        
        # 6. Homography 계산
        self._print_debug("Step 6: Computing homography...")
        self.homography_matrix = self.compute_homography(corners_2d, target_corners)
        
        # 7. RGB 이미지와 마스크 모두 원근 변환 적용
        self._print_debug("Step 7: Applying perspective transformation to both image and mask...")
        rectified_image = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 마스크도 동일하게 변환
        rectified_mask = cv2.warpPerspective(
            segmentation_mask.astype(np.uint8), 
            self.homography_matrix, 
            output_size,
            flags=cv2.INTER_NEAREST,  # 마스크는 nearest neighbor 사용
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 8. 순수한 벽 텍스처 추출
        self._print_debug("Step 8: Extracting pure wall texture...")
        wall_texture, texture_bbox = self.extract_wall_texture_from_rectified(
            rectified_image, rectified_mask
        )
        
        if self.save_intermediate:
            self._save_debug_image(wall_texture, "wall_texture.png")
            
        if self.visualize:
            self._visualize_mbr_rectification_result(
                image, rectified_image, corners_3d, corners_2d, target_corners, 
                wall_mask, wall_normal, camera_intrinsic
            )
        
        return (rectified_image, rectified_mask, wall_texture, texture_bbox, 
                self.homography_matrix, corners_3d, corners_2d, target_corners, wall_normal)

    def process_multiple_walls_with_mbr(self, step1_output_path, wall_index=0, is_floor=False):
        """
        MBR 방식을 사용하여 제공된 데이터 구조에서 여러 벽을 처리하는 메인 함수 (floor normal 지원)
        """
        # 데이터 로드 (floor normal 포함)
        image, point_cloud, wall_masks, step1_output_info, floor_normal = self.load_data_from_paths(step1_output_path, is_floor=is_floor)
        
        if wall_index >= len(wall_masks):
            raise ValueError(f"Wall index {wall_index} is out of range. Available walls: {len(wall_masks)}")
        
        # 선택된 벽 처리
        selected_mask = wall_masks[wall_index]
        camera_intrinsic = step1_output_info["K"]
        
        if self.debug_print:
            self._print_debug(f"Processing wall {wall_index} with mask shape: {selected_mask.shape}")
            self._print_debug(f"Camera intrinsic matrix:\n{np.array(camera_intrinsic)}")
            if floor_normal is not None:
                self._print_debug(f"Using floor normal: {floor_normal}")
            else:
                self._print_debug("Floor normal not available - using camera coordinate assumption")
        
        # MBR 기반 텍스처 복원 실행 (floor normal 전달)
        if is_floor:
            (rectified_image, rectified_mask, wall_texture, texture_bbox,
            homography, corners_3d, corners_2d, target_corners, wall_normal) = self.rectify_texture_with_mbr_floor(
                image, selected_mask, point_cloud, camera_intrinsic
            )
        else:
            (rectified_image, rectified_mask, wall_texture, texture_bbox,
            homography, corners_3d, corners_2d, target_corners, wall_normal) = self.rectify_texture_with_mbr(
                image, selected_mask, point_cloud, camera_intrinsic, floor_normal=floor_normal
            )

        return {
            'rectified_image': rectified_image,
            'rectified_mask': rectified_mask,
            'wall_texture': wall_texture,
            'texture_bbox': texture_bbox,
            'homography': homography,
            'corners_3d': corners_3d,
            'corners_2d': corners_2d,
            'target_corners': target_corners,
            'wall_normal': wall_normal,
            'floor_normal': floor_normal,
            'original_image': image,
            'wall_mask': selected_mask,
            'camera_intrinsic': camera_intrinsic
        }
        
    def _visualize_mbr_rectification_result(self, original_image, rectified_image, 
                                          corners_3d, corners_2d, target_corners, 
                                          wall_mask, wall_normal, camera_intrinsic):
        """
        MBR 기반 rectification 결과 시각화
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 원본 이미지와 투영된 3D 모서리점들
        axes[0, 0].imshow(original_image)
        axes[0, 0].scatter(corners_2d[:, 0], corners_2d[:, 1], c='red', s=200, marker='s', 
                          edgecolors='yellow', linewidth=3, label='Projected 3D Corners')
        
        # 모서리점들을 선으로 연결
        for i in range(4):
            next_i = (i + 1) % 4
            axes[0, 0].plot([corners_2d[i, 0], corners_2d[next_i, 0]], 
                           [corners_2d[i, 1], corners_2d[next_i, 1]], 
                           'r-', linewidth=3, alpha=0.7)
        
        # 모서리점 번호 표시
        for i, point in enumerate(corners_2d):
            axes[0, 0].annotate(f'{i}', (point[0], point[1]), xytext=(8, 8), 
                              textcoords='offset points', color='yellow', 
                              fontweight='bold', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        axes[0, 0].set_title('Original Image with Projected 3D MBR Corners')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # 2. 벽 마스크
        axes[0, 1].imshow(wall_mask, cmap='gray')
        axes[0, 1].set_title('Wall Segmentation Mask')
        axes[0, 1].axis('off')
        
        # 3. 타겟 직사각형 시각화
        target_img = np.zeros((512, 512, 3), dtype=np.uint8)
        target_corners_int = target_corners.astype(int)
        cv2.fillPoly(target_img, [target_corners_int], (100, 100, 100))
        
        axes[0, 2].imshow(target_img)
        axes[0, 2].scatter(target_corners[:, 0], target_corners[:, 1], c='green', s=200, 
                          marker='s', edgecolors='yellow', linewidth=3, label='Target Corners')
        
        # 타겟 모서리점들을 선으로 연결
        for i in range(4):
            next_i = (i + 1) % 4
            axes[0, 2].plot([target_corners[i, 0], target_corners[next_i, 0]], 
                           [target_corners[i, 1], target_corners[next_i, 1]], 
                           'g-', linewidth=3)
        
        # 타겟 모서리점 번호 표시
        for i, point in enumerate(target_corners):
            axes[0, 2].annotate(f'{i}', (point[0], point[1]), xytext=(8, 8), 
                              textcoords='offset points', color='yellow', 
                              fontweight='bold', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
        
        axes[0, 2].set_title('Target Rectangle Layout')
        axes[0, 2].legend()
        axes[0, 2].axis('off')
        
        # 4. 복원된 텍스처
        axes[1, 0].imshow(rectified_image)
        axes[1, 0].set_title('MBR-based Rectified Wall Texture')
        axes[1, 0].axis('off')
        
        # 5. 3D MBR 시각화 (간단 버전)
        ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
        
        # 3D 모서리점들
        ax_3d.scatter(corners_3d[:, 0], corners_3d[:, 1], corners_3d[:, 2], 
                     color='red', s=200, marker='s', label='3D MBR Corners')
        
        # 3D MBR 경계선
        for i in range(4):
            next_i = (i + 1) % 4
            ax_3d.plot([corners_3d[i, 0], corners_3d[next_i, 0]], 
                      [corners_3d[i, 1], corners_3d[next_i, 1]], 
                      [corners_3d[i, 2], corners_3d[next_i, 2]], 
                      'r-', linewidth=3)
        
        # 모서리점 번호 표시
        for i, corner in enumerate(corners_3d):
            ax_3d.text(corner[0], corner[1], corner[2], f'  {i}', 
                      fontsize=10, fontweight='bold')
        
        ax_3d.set_title('3D MBR in Camera Space')
        ax_3d.legend()
        
        # 6. 변환 품질 분석
        axes[1, 2].axis('off')
        
        # 변환 통계 계산
        mbr_width_3d = np.linalg.norm(corners_3d[1] - corners_3d[0])
        mbr_height_3d = np.linalg.norm(corners_3d[3] - corners_3d[0])
        target_width = np.linalg.norm(target_corners[1] - target_corners[0])
        target_height = np.linalg.norm(target_corners[3] - target_corners[0])
        
        # 투영 정확도 검사 (모서리점들이 이미지 경계 내에 있는지)
        img_h, img_w = original_image.shape[:2]
        x_in_bounds = (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < img_w)
        y_in_bounds = (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < img_h)
        corners_in_bounds = np.all(x_in_bounds & y_in_bounds)
        
        stats_text = f"""
MBR Rectification Analysis:
═══════════════════════════

3D Measurements:
─────────────────
Width: {mbr_width_3d:.3f} m
Height: {mbr_height_3d:.3f} m
Aspect Ratio: {mbr_width_3d/mbr_height_3d:.3f}

2D Target:
─────────────────
Width: {target_width:.0f} px
Height: {target_height:.0f} px
Aspect Ratio: {target_width/target_height:.3f}

Projection Quality:
─────────────────
All corners in image: {'✓' if corners_in_bounds else '✗'}
Homography rank: {'✓' if self.homography_matrix is not None else '✗'}

Corner Coordinates (2D):
─────────────────
0: ({corners_2d[0,0]:.1f}, {corners_2d[0,1]:.1f})
1: ({corners_2d[1,0]:.1f}, {corners_2d[1,1]:.1f})
2: ({corners_2d[2,0]:.1f}, {corners_2d[2,1]:.1f})
3: ({corners_2d[3,0]:.1f}, {corners_2d[3,1]:.1f})
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('MBR-based Wall Texture Rectification Results', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # 추가 정보 출력
        if self.debug_print:
            self._print_debug("\n=== MBR Rectification Summary ===")
            self._print_debug(f"3D Wall dimensions: {mbr_width_3d:.3f}m × {mbr_height_3d:.3f}m")
            self._print_debug(f"2D Target dimensions: {target_width:.0f}px × {target_height:.0f}px")
            self._print_debug(f"Projection status: {'All corners visible' if corners_in_bounds else 'Some corners outside image'}")
            self._print_debug(f"Homography computed: {'Successfully' if self.homography_matrix is not None else 'Failed'}")

    # 편의 함수들 (클래스 내부로 이동)
    def process_single_wall_enhanced_mbr(self, step1_output_path, wall_index=0):
        """
        Enhanced MBR 방법으로 단일 벽을 처리하는 편의 함수 (텍스처 추출 포함)
        """
        return self.process_multiple_walls_with_mbr(step1_output_path, wall_index)

    def process_all_walls_enhanced_mbr(self, step1_output_path):
        """
        Enhanced MBR 방법으로 모든 벽을 배치 처리하는 함수 (텍스처 추출 포함)
        """
        results = []
        
        # 데이터 로드하여 벽 개수 확인
        image, point_cloud, wall_masks, step1_output_info, floor_normal = self.load_data_from_paths(step1_output_path)
        
        self._print_debug(f"총 {len(wall_masks)}개의 벽을 Enhanced MBR 방식으로 처리합니다.")
        
        for wall_idx in range(len(wall_masks)):
            try:
                self._print_debug(f"\n--- Processing Wall {wall_idx} ---")
                result = self.process_multiple_walls_with_mbr(step1_output_path, wall_idx)
                results.append(result)
                if self.debug_print:
                    self._print_debug(f"Wall {wall_idx} 처리 완료")
                    self._print_debug(f"  - Rectified image: {result['rectified_image'].shape}")
                    self._print_debug(f"  - Pure texture: {result['wall_texture'].shape}")
                    texture_coverage = result['wall_texture'].shape[0]*result['wall_texture'].shape[1] / (result['rectified_image'].shape[0]*result['rectified_image'].shape[1])*100
                    self._print_debug(f"  - Texture coverage: {texture_coverage:.1f}%")
            except Exception as e:
                self._print_debug(f"Wall {wall_idx} 처리 중 오류: {e}")
                results.append(None)
        
        return results

    def save_wall_textures(self, results, output_dir):
        """
        처리된 벽 텍스처들을 파일로 저장하는 유틸리티 함수
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            if result is not None:
                # Rectified 이미지 저장
                rectified_pil = Image.fromarray(result['rectified_image'])
                rectified_pil.save(os.path.join(output_dir, f"wall_{i}_rectified.png"))
                
                # 순수 벽 텍스처 저장
                texture_pil = Image.fromarray(result['wall_texture'])
                texture_pil.save(os.path.join(output_dir, f"wall_{i}_texture.png"))
                
                if self.debug_print:
                    self._print_debug(f"Wall {i} 텍스처 저장 완료:")
                    self._print_debug(f"  - Rectified: wall_{i}_rectified.png")
                    self._print_debug(f"  - Pure texture: wall_{i}_texture.png")
