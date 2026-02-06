# Standard Libraries
import os
import json
from pathlib import Path
from copy import deepcopy

# Third-party Libraries
import torch as th
import numpy as np
from PIL import Image
import imageio
from loguru import logger as log

# OmniGibson Libraries
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn

# Local / Project Utilities
from our_method.utils.processing_utils import prepare_output_dir, NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from our_method.utils.scene_utils import create_scene, take_photo, compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory
import our_method.utils.transform_utils as T


class RealSceneGenerator:
    """
    3rd Step in ACDC pipeline. This takes in the output from Step 2 (Digital Cousin Matching) and generates
    fully populated digital cousin scenes

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 2, which includes the following:
            - Per-object (category,, model, pose) digital cousin information

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """
    
    # Set of non-collidable categories
    NON_COLLIDABLE_CATEGORIES = {
        "towel",
        "rug",
        "mirror",
        "picture",
        "painting",
        "window",
        "art",
    }

    CATEGORIES_MUST_ON_FLOOR = {
        "rug",
        "carpet"
    }
    
    SAMPLING_METHODS = {
        "random",
        "ordered",
    }

    def __init__(
            self,
            verbose=False,
    ):
        """
        Args:
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.verbose = verbose
        # Instance variables to be set in __call__ or helper methods
        self.n_scenes = 0
        self.sampling_method = "ordered"
        self.resolve_collision = True
        self.discard_objs = None
        self.save_dir = None
        self.visualize_scene = False
        self.visualize_scene_tilt_angle = 0
        self.visualize_scene_radius = 5
        self.save_visualization = True

        self.step_1_output_path = None
        self.step_2_output_path = None
        self.step_2_output_info = None
        
        # Loaded data from Step 1 & 2
        self.n_cousins = 0
        self.n_objects = 0
        self.cousins = {}
        self.rgb = None
        self.h = 0
        self.w = 0
        self.K = None
        self.z_dir = None
        self.wall_mask_planes = None
        self.cam_pos = None
        self.cam_quat = None
        self.pc = None
        self.detected_categories = None


    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            camera_info=None,
            n_scenes=1,
            sampling_method="ordered",
            resolve_collision=True,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=1,
            save_visualization=True,
            save_camera_info_extrinsic=False
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:
        ...
        """
        # Store all input parameters as instance variables
        self.step_1_output_path = step_1_output_path
        self.step_2_output_path = step_2_output_path
        self.camera_info = camera_info
        self.n_scenes = n_scenes
        self.sampling_method = sampling_method
        self.resolve_collision = resolve_collision
        self.visualize_scene = visualize_scene
        self.visualize_scene_tilt_angle = visualize_scene_tilt_angle
        self.visualize_scene_radius = visualize_scene_radius
        self.save_visualization = save_visualization
        self.save_camera_info_extrinsic = save_camera_info_extrinsic

        # Load step 2 info
        with open(self.step_2_output_path, "r") as f:
            self.step_2_output_info = json.load(f)
        
        # Load relevant information from prior steps
        self.n_cousins = self.step_2_output_info["metadata"]["n_cousins"]
        self.cousins = self.step_2_output_info["objects"]
        self.n_objects = self.step_2_output_info["metadata"]["n_objects"]

        # Sanity check
        assert self.sampling_method in self.SAMPLING_METHODS, \
            f"Got invalid sampling_method! Valid methods: {self.SAMPLING_METHODS}, got: {self.sampling_method}"
        
        # Setup save directory and discard list
        self.save_dir = prepare_output_dir(self.step_2_output_path, save_dir, "step_3_output")
        if discard_objs:
            self.discard_objs = set(discard_objs.split(","))

        if self.verbose:
            log.info(f"Generating simulated scenes given output {self.step_2_output_path}...")
            log.debug("Generating simulated scenes in OmniGibson")

        # Load input data and compute 3D context
        self._load_and_setup_environment()

        # Launch omnigibson
        og.launch()

        # Loop over all sample indices to generate individual scenes
        for scene_count in range(self.n_scenes):
            log.info(f"[Scene {scene_count + 1} / {self.n_scenes}]")

            scene_save_dir = f"{self.save_dir}/scene_{scene_count}"
            Path(scene_save_dir).mkdir(parents=True, exist_ok=True)
   
            # Determine which cousin model to use
            cousin_idxs = self._select_cousin_indices(scene_count)
            
            # --- 1. Initial Object Processing and Alignment ---
            scene_info = self._process_individual_objects(
                scene_count=scene_count,
                scene_save_dir=scene_save_dir,
                cousin_idxs=cousin_idxs,
            )

            # --- 2. Scene Refinement, Collision, and Final Placement ---
            self._refine_scene_and_resolve_physics(
                scene_count=scene_count,
                scene_save_dir=scene_save_dir,
                scene_info=scene_info,
            )

        # Compile final results across all scenes and save to disk
        step_3_output_path = self._save_step3_outputs()

        log.success("Completed Simulated Scene Generation!")

        return True, step_3_output_path

    # --- Private Helper Methods ---

    def _load_and_setup_environment(self):
        """Loads input data, computes point cloud, and camera pose."""
        with open(self.step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            self.detected_categories = json.load(f)

        seg_dir = self.detected_categories["segmentation_dir"]
        self.K = np.array(step_1_output_info["K"])
        self.rgb = np.array(Image.open(step_1_output_info["input_rgb"]))
        raw_depth = np.array(Image.open(step_1_output_info["input_depth"]))
        depth_limits = np.array(step_1_output_info["depth_limits"])
        depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
        self.pc = compute_point_cloud_from_depth(depth=depth, K=self.K)
        self.h, self.w, _ = self.rgb.shape

        self.z_dir = np.array(step_1_output_info["z_direction"])
        self.wall_mask_planes = step_1_output_info["wall_mask_planes"]
        origin_pos = np.array(step_1_output_info["origin_pos"])
        if self.save_camera_info_extrinsic and self.camera_info is not None:
            self.w = self.camera_info['intrinsics']['image_width']
            self.h = self.camera_info['intrinsics']['image_height']
            self.cam_pos = self.camera_info['camera']['position']
            self.cam_quat = self.camera_info['camera']['orientation']
        else:
            self.cam_pos, self.cam_quat = compute_relative_cam_pose_from(z_dir=self.z_dir, origin_pos=origin_pos)

    def _select_cousin_indices(self, scene_count):
        """Determines the cousin index for each object based on the sampling method."""
        if self.sampling_method == "random":
            cousin_idxs = dict()
            for obj_name in self.cousins.keys():
                cousin_idxs[obj_name] = np.random.randint(0, self.n_cousins)
        elif self.sampling_method == "ordered":
            cousin_idxs = {obj_name: scene_count for obj_name in self.cousins.keys()}
        else:
            raise ValueError(f"sampling_method {self.sampling_method} not supported!")
        return cousin_idxs

    def _process_individual_objects(
        self,
        scene_count, scene_save_dir, cousin_idxs,
    ):
        """Loads and aligns each object individually, saving per-object scene info."""
        scene = create_scene(floor=False)
        og.sim.viewer_camera.image_width = self.w
        og.sim.viewer_camera.image_height = self.h
        og.sim.viewer_camera.set_position_orientation(th.tensor(self.cam_pos, dtype=th.float), th.tensor(self.cam_quat, dtype=th.float))
        
        seg_dir = self.detected_categories["segmentation_dir"]

        for obj_idx, (obj_name, obj_cousin_idx) in enumerate(cousin_idxs.items()):
            if self.verbose:
                log.info(f"[Scene {scene_count + 1} / {self.n_scenes}] [Object {obj_idx + 1} / {self.n_objects}] generating...")
            if self.discard_objs and obj_name in self.discard_objs:
                continue

            obj_info = self.step_2_output_info["objects"][obj_name]
            is_articulated = obj_info["articulated"]
            
            obj_mask = np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected_mask_pruned.png"))
            pc_obj = self.pc.reshape(-1, 3)[np.array(obj_mask).flatten().nonzero()[0]]

            cousin_info = obj_info["cousins"][obj_cousin_idx]

            # Import the cousin asset
            with og.sim.stopped():
                obj = DatasetObject(
                    name=obj_name, category=cousin_info["category"], model=cousin_info["model"], visual_only=True
                )
                scene.add_object(obj)
            og.sim.step()

            # Determine the reprojection offset
            pan_angle_offset, _ = get_reproject_offset(
                pc_obj=deepcopy(pc_obj), z_dir=self.z_dir, xy_dist=2.30, z_dist=0.65
            )

            take_photo(n_render_steps=50)
            # Align object model to point cloud
            obj_scale, obj_bbox_extent, tf_from_cam = align_model_pose(
                obj=obj, pc_obj=pc_obj, obj_z_angle=cousin_info["z_angle"] + pan_angle_offset,
                obj_ori_offset=cousin_info["ori_offset"], z_dir=deepcopy(self.z_dir),
                cam_pos=self.cam_pos, cam_quat=self.cam_quat, is_articulated=is_articulated, verbose=self.verbose,
            )
            
            take_photo(n_render_steps=50)
            wall_mount_fpaths = self.detected_categories["mount"][obj_idx]["wall"]
            if wall_mount_fpaths is not None:
                # Align object with the wall
                for mount_wall_idx, wall_mount_fpath in enumerate(wall_mount_fpaths):
                    obj_scale, obj_bbox_extent, tf_from_cam = align_obj_with_wall(
                        obj=obj, cam_pos=self.cam_pos, cam_quat=self.cam_quat,
                        wall_normal=self.wall_mask_planes[wall_mount_fpath]["normal"],
                        wall_point=self.wall_mask_planes[wall_mount_fpath]["point"],
                        wall_is_vertical=True, resize_only=mount_wall_idx > 0,
                    )
            take_photo(n_render_steps=100)

            # Save per-object scene info
            obj_save_dir = f"{scene_save_dir}/{obj_name}"
            Path(obj_save_dir).mkdir(parents=True, exist_ok=True)
            obj_scene_info = {
                "category": obj.category, "model": obj.model, "scale": obj_scale,
                "bbox_extent": obj_bbox_extent, "tf_from_cam": tf_from_cam,
                "mount": self.detected_categories["mount"][obj_idx],
            }
            with open(f"{obj_save_dir}/{obj_name}_scene_info.json", "w+") as f:
                json.dump(obj_scene_info, f, indent=4, cls=NumpyTorchEncoder)

            take_photo()
            scene.remove_object(obj)

        # Compile and return scene info dictionary
        scene_info = {"resolution": [self.h, self.w], "cam_pose": [self.cam_pos, self.cam_quat], "objects": {}}
        for obj_name in cousin_idxs.keys():
            if self.discard_objs and obj_name in self.discard_objs:
                continue
            with open(f"{scene_save_dir}/{obj_name}/{obj_name}_scene_info.json", "r") as f:
                scene_obj_info = json.load(f)
            scene_info["objects"][obj_name] = scene_obj_info
        
        scene_info["scene_graph"] = f"{scene_save_dir}/scene_{scene_count}_graph.json"

        return scene_info

    def _refine_scene_and_resolve_physics(
        self,
        scene_count, scene_save_dir, scene_info,
    ):
        """Loads full scene, computes scene graph, resolves collisions, and visualizes."""
        scene = RealSceneGenerator.load_cousin_scene(scene_info=scene_info, visual_only=True)
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {scene_info['resolution'][0]}] refining scene graph...")

        # --- 1. Infer Scene Graph and Adjust Height (Z-offset) ---
        scene_graph_info, all_obj_bbox_info, sorted_z_obj_bbox_info, final_scene_info = \
            self._infer_scene_graph_and_adjust_height(scene, scene_info)
        
        # Save scene graph
        scene_graph_info_path = f"{scene_save_dir}/scene_{scene_count}_graph.json"
        with open(scene_graph_info_path, "w+") as f:
            json.dump(scene_graph_info, f, indent=4, cls=NumpyTorchEncoder)

        # --- 2. Collision Resolution (X/Y plane) ---
        sorted_x_obj_bbox_info = dict(sorted(sorted_z_obj_bbox_info.items(), key=lambda x: x[1]['lower'][0], reverse=True))
        obj_names = list(sorted_x_obj_bbox_info.keys())
        
        if self.resolve_collision:
            self._resolve_horizontal_collisions(scene_count, scene, obj_names, scene_graph_info, final_scene_info)
        else:
            if self.verbose:
                log.info(f"[Scene {scene_count + 1} / {1}] skip depenetrating collisions.")

        # --- 3. Final Placement (Vertical Drop) ---
        self._resolve_vertical_placement(scene_count, scene, obj_names, scene_graph_info, all_obj_bbox_info, final_scene_info)

        # Take final physics step, then save visualization + info
        og.sim.step_physics()
        
        # --- 4. Save Final Visualization and Info ---
        self._save_final_outputs(scene_count, scene_save_dir, final_scene_info)
        
        if self.visualize_scene:
            self._visualize_scene_video(
                scene, scene_save_dir
            )
        
        # Return final info (though it's saved to disk)
        return final_scene_info

    def _infer_scene_graph_and_adjust_height(self, scene, scene_info):
        """Infers object relationships and adjusts object height for vertical plausibility."""
        all_obj_bbox_info = {}
        for obj_name, obj_info in scene_info["objects"].items():
            if self.discard_objs and obj_name in self.discard_objs:
                continue
            obj = scene.object_registry("name", obj_name)
            obj_bbox_info = compute_obj_bbox_info(obj=obj)
            obj_bbox_info["articulated"] = self.step_2_output_info["objects"][obj_name]["articulated"]
            obj_bbox_info["mount"] = obj_info["mount"]
            all_obj_bbox_info[obj_name] = obj_bbox_info
        sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: x[1]['lower'][2]))

        scene_graph_info = {"floor": {"objOnTop": [], "objBeneath": None, "mount": {"floor": True, "wall": False}}}
        final_scene_info = deepcopy(scene_info)

        for name in sorted_z_obj_bbox_info:
            obj_name_beneath, z_offset = compute_object_z_offset(
                target_obj_name=name, sorted_obj_bbox_info=sorted_z_obj_bbox_info, verbose=self.verbose,
            )
            obj = scene.object_registry("name", name)

            if scene_info["objects"][name]["category"] in self.CATEGORIES_MUST_ON_FLOOR:
                obj_name_beneath = "floor"
                z_offset = -sorted_z_obj_bbox_info[name]["lower"][-1]

            # Update Scene Graph
            scene_graph_info.setdefault(name, {"objOnTop": [], "objBeneath": obj_name_beneath, "mount": None})
            scene_graph_info.setdefault(obj_name_beneath, {"objOnTop": [], "objBeneath": None, "mount": None})
            scene_graph_info[name]["objBeneath"] = obj_name_beneath
            scene_graph_info[obj_name_beneath]["objOnTop"].append(name)
            scene_graph_info[name]["mount"] = scene_info["objects"][name]["mount"]
            obj.keep_still()

            # Apply Z-offset
            if z_offset != 0:
                mount_type = scene_info["objects"][name]["mount"]
                if not mount_type["floor"] and z_offset <= 0: continue
                new_center = sorted_z_obj_bbox_info[name]["center"] + np.array([0.0, 0.0, z_offset])
                obj.set_bbox_center_position_orientation(position=th.tensor(new_center, dtype=th.float), orientation=None)
                og.sim.step_physics()
                sorted_z_obj_bbox_info[name].update(compute_obj_bbox_info(obj=obj))

            # Update relative transformation
            obj_pos, obj_quat = obj.get_position_orientation()
            rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
            final_scene_info["objects"][name]["tf_from_cam"] = T.pose2mat(rel_tf)

        for obj in scene.objects: obj.keep_still()
        og.sim.step_physics()
        return scene_graph_info, all_obj_bbox_info, sorted_z_obj_bbox_info, final_scene_info

    def _resolve_horizontal_collisions(self, scene_count, scene, obj_names, scene_graph_info, final_scene_info):
        """Resolves collisions between objects in the X/Y plane by moving them apart."""
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {1}] depenetrating collisions...")

        for obj1_idx, obj1_name in enumerate(obj_names):
            if any(cat in obj1_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue

            obj1 = scene.object_registry("name", obj1_name)
            obj1.keep_still()
            obj1.visual_only = False

            for obj2_name in obj_names[obj1_idx + 1:]:
                if any(cat in obj2_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue
                assert obj1_name != obj2_name

                if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                    continue
                
                obj2 = scene.object_registry("name", obj2_name)
                old_state = og.sim.dump_state()
                obj2.keep_still()
                obj2.visual_only = False
                og.sim.step_physics()

                if obj2.states[Touching].get_value(obj1):
                    if self.verbose:
                        log.info(f"Detected collision between {obj1_name} and {obj2_name}")
                    
                    obj2_ori_mat = T.quat2mat(obj2.get_position_orientation()[1].cpu().detach().numpy())
                    obj2_x_dir = obj2_ori_mat[:, 0]
                    obj2_y_dir = obj2_ori_mat[:, 1]
                    center_step_size = 0.01
                    obj2_to_obj1 = (obj1.get_position_orientation()[0] - obj2.get_position_orientation()[0]).cpu().detach().numpy()

                    chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                    center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis

                    while obj2.states[Touching].get_value(obj1):
                        og.sim.load_state(old_state)
                        new_center = obj2.get_position_orientation()[0] + th.tensor(center_step_dir, dtype=th.float) * center_step_size
                        obj2.set_position_orientation(position=new_center)
                        old_state = og.sim.dump_state()
                        og.sim.step_physics()

                    og.sim.load_state(old_state)
                    obj2.set_position_orientation(position=new_center)
                    obj_pos, obj_quat = obj2.get_position_orientation()
                    rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
                    final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                else:
                    og.sim.load_state(old_state)
                obj2.visual_only = True
            obj1.visual_only = True

    def _resolve_vertical_placement(self, scene_count, scene, obj_names, scene_graph_info, all_obj_bbox_info, final_scene_info):
        """Uses physics steps to gently place objects onto the surface beneath them."""
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {1}] placing all objects down...")

        for obj in scene.objects: obj.keep_still()
        og.sim.step_physics()
        
        for obj1_name in obj_names:
            if any(cat in obj1_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue
            
            if scene_graph_info[obj1_name]['objBeneath'] == "floor" or not all_obj_bbox_info[obj1_name]["mount"]["floor"]:
                continue
            
            obj_beneath_name = scene_graph_info[obj1_name]["objBeneath"]
            obj_beneath = scene.object_registry("name", obj_beneath_name)

            if "no_top" in obj_beneath.category or any(cat in obj_beneath_name for cat in self.NON_COLLIDABLE_CATEGORIES):
                continue
            
            obj1 = scene.object_registry("name", obj1_name)
            obj_beneath.keep_still()
            obj_beneath.visual_only = False
            old_state = og.sim.dump_state()

            obj1.keep_still()
            obj1.visual_only = False
            obj1_lower_corner, _ = obj1.aabb
            obj1_low_z = obj1_lower_corner[-1].item()
            obj_beneath_lower_corner, _ = obj_beneath.aabb
            obj_beneath_low_z = obj_beneath_lower_corner[-1].item()
            center_step_size = 0.005
            og.sim.step_physics()

            if not obj1.states[Touching].get_value(obj_beneath):
                while obj1_low_z >= max(0, obj_beneath_low_z) and \
                    not obj1.states[Touching].get_value(obj_beneath):
                    og.sim.load_state(old_state)
                    new_center = obj1.get_position_orientation()[0] + th.tensor([0, 0, -1.0]) * center_step_size
                    obj1_low_z -= center_step_size
                    obj1.set_position_orientation(position=new_center)
                    old_state = og.sim.dump_state()
                    og.sim.step_physics()

                og.sim.load_state(old_state)
                final_position = obj1.get_position_orientation()[0] - th.tensor([0, 0, -1.0]) * center_step_size
                obj1.set_position_orientation(position=final_position)
                obj_pos, obj_quat = obj1.get_position_orientation()
                rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
                final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)
            else:
                og.sim.load_state(old_state)

            obj_beneath.keep_still()
            obj1.keep_still()
            og.sim.step_physics()
            obj1.visual_only = True
            obj_beneath.visual_only = True

    def _save_final_outputs(self, scene_count, scene_save_dir, final_scene_info):
        """Saves the final RGB visualization and scene info JSON."""
        scene_rgb = take_photo(n_render_steps=10)
        H, W, _ = scene_rgb.shape
        resized_rgb = resize_image(self.rgb, height=H)
        concat_scene_rgb = np.concatenate([scene_rgb[:, :, :3], scene_rgb[:,:,:3]], axis=1)
        Image.fromarray(concat_scene_rgb).save(f"{scene_save_dir}/scene_{scene_count}_visualization.png")

        with open(f"{scene_save_dir}/scene_{scene_count}_info.json", "w+") as f:
            json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)

    def _save_step3_outputs(self) -> str:
        """
        Compiles and saves the final Step 3 output JSON across all generated scenes.
        """
        step_3_output_info = dict()
        for scene_count in range(self.n_scenes):
            scene_name = f"scene_{scene_count}"
            final_scene_info_path = f"{self.save_dir}/{scene_name}/{scene_name}_info.json"
            with open(final_scene_info_path, "r") as f:
                final_scene_info = json.load(f)
            step_3_output_info[scene_name] = final_scene_info

        step_3_output_path = f"{self.save_dir}/step_3_output_info.json"
        with open(step_3_output_path, "w+") as f:
            json.dump(step_3_output_info, f, indent=4, cls=NumpyTorchEncoder)

        return step_3_output_path


    def _visualize_scene_video(self, scene, scene_save_dir):
        """Generates and saves a rotating video visualization of the final scene."""
        og.sim.viewer_camera.add_modality('seg_semantic')
        aabb_points = []
        for obj in scene.objects:
            p1, p2 = obj.aabb
            aabb_points.append(p1)
            aabb_points.append(p2)
            if ToggledOn in obj.states:
                obj.states[ToggledOn].link.visible = False

        min_x = min([p[0] for p in aabb_points])
        min_y = min([p[1] for p in aabb_points])
        max_x = max([p[0] for p in aabb_points])
        max_y = max([p[1] for p in aabb_points])
        vis_cam_pos, vis_cam_ori = og.sim.viewer_camera.get_position_orientation()
        vis_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, vis_cam_pos[-1])
        cam_commands = get_vis_cam_trajectory(
            center_pos=vis_center, cam_pos=vis_cam_pos, cam_quat=vis_cam_ori,
            d_tilt=self.visualize_scene_tilt_angle, radius=self.visualize_scene_radius, n_steps=100
        )

        for _ in range(50):
            og.sim.render()

        if self.save_visualization:
            video_path = f"{scene_save_dir}/visualization_video.mp4"
            img_dir = f"{scene_save_dir}/scene_visualization"
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=20)
        for i, (pos, quat) in enumerate(cam_commands):
            og.sim.viewer_camera.set_position_orientation(pos, quat)
            og.sim.render()
            if self.save_visualization:
                obs, obs_info = og.sim.viewer_camera.get_obs()
                vis_rgb = obs["rgb"].cpu().detach().numpy()
                seg_semantic = obs["seg_semantic"].cpu().detach().numpy()

                filter_names = {"floors", "background"}
                filter_ids = {idn for idn, name in obs_info["seg_semantic"].items() if name in filter_names}
                seg_mask = np.ones_like(seg_semantic).astype(np.uint8) * 255
                for filter_id in filter_ids:
                    seg_mask[np.where(seg_semantic == filter_id)] = 0
                masked_vis_rgb = vis_rgb.astype(np.uint8)
                masked_vis_rgb[seg_mask == 0] = [0, 0, 0, 1]
                video_writer.append_data(masked_vis_rgb)
                
                vis_rgb[:, :, 3] = seg_mask
                Image.fromarray(vis_rgb).save(f"{img_dir}/vis_frame_{i}.png")
        if self.save_visualization:
            video_writer.close()

    # --- Static Helper Methods ---

    @staticmethod
    def load_cousin_scene(scene_info, visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath
        ...
        """
        # Stop sim, clear it, then load empty scene
        scene = create_scene(floor=True)

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(scene_info)

        # Set viewer camera to proper pose
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))

        # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
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
                obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))
        
        # Initialize all objects by taking one step
        og.sim.step()
        return scene

    def joint_test(self, scene, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera
        ...
        """
        obj = scene.object_registry("name", "cabinet_0") 

        joint_index = 0
        positions = obj.get_joint_positions()

        for step in range(n_render_steps):
            # Change joint position every 10 steps (0 <-> 1.5)
            if step % 10 == 0:
                positions[joint_index] = 1.5 if positions[joint_index] == 0 else 0
                obj.set_joint_positions(positions)  # Update

            # Run physics simulation and render
            og.sim.step_physics()
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        return rgb