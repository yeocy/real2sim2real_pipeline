import torch as th
import numpy as np
from pathlib import Path
from PIL import Image
from copy import deepcopy
import os
import json
import imageio
import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn
import digital_cousins
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from digital_cousins.utils.scene_utils import compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory
import digital_cousins.utils.transform_utils as T

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

class VisualizeScene:
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

    def __call__(
            self,
            scene_info_path,
            n_scenes=1,
            sampling_method="random",
            resolve_collision=True,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=5,
            save_visualization=True
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:

        1. Compute camera pose and world origin point from step 1 output.
        2. Separately set each object in correct position and orientation w.r.t. the viewer camera,
           and save the relative transformation between the object and the camera.
        3. Put all objects in a single scene.
        4. Infer objects OnTop relationship. We currently only support OnTop cross-object relationship, so there might
            be artifacts if an object is 'In' another object, like books in a bookshelf.
        5. Process collisions and put objects onto the floor or objects beneath to generate a physically plausible scene.
        6. (Optionally) visualize the reconstructed scene.

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            step_2_output_path (str): Absolute path to the output file generated from Step 2 (DigitalCousinMatcher)
            n_scenes (int): Number of scenes to generate. This number cannot be greater than the number of cousins
                generated from Step 2 if @sampling_method="ordered" or greater than the product of all possible cousin
                combinations if @sampling_method="random"
            sampling_method (str): Sampling method to use when generating scenes. "random" will randomly select a cousin
                for each detected object in Step 1 (total combinations: N_cousins ^ N_objects). "ordered" will
                sequentially iterate over each detected object and generate scenes with corresponding ordered cousins,
                i.e.: a scene with all 1st cousins, a scene with all 2nd cousins, etc. (total combinations: N_cousins).
                Note that in both cases, the first scene generated will always be composed of all the closest (first)
                cousins. Default is "random"
            resolve_collision (bool): Whether to depenetrate collisions. When the point cloud is not denoised properly,
                or the mounting type is wrong, the object can be unreasonably large. Or when two objects in the input image
                intersect with each other, we may move an object by a non-trivial distance to depenetrate collision, so
                objects on top may fall down to the floor, and other objects may also need to be moved to avoid collision
                with this object. Under both cases, we recommend setting @resolve_collision to False to visualize the
                raw output.
            discard_objs (str): Names of objects to discard during reconstruction, seperated by comma, i.e., obj_1,obj_2,obj_3.
                Do not add space between object names.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_2_output_path
            visualize_scene (bool): Whether to visualize the scene after reconstruction. If True, the viewer camera will
                rotate around the scene's center point with a @visualize_scene_tilt_angle tilt cangle, and a 
                @visualize_scene_radius radius.
            visualize_scene_tilt_angle (float): The camera tilt angle in degree when visualizing the reconstructed scene. 
                This parameter is only used when @visualize_scene is set to True
            visualize_scene_radius (float): The camera rotating raiud in meters when visualizing the reconstructed scene.
                This parameter is only used when @visualize_scene is set to True
            save_visualization (bool): Whether to save the visualization results. This parameter is only used when 
                @visualize_scene is set to True

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """

        if self.verbose:
            print("""

####################################################
####  Generating simulated scenes in OmniGibson ####
####################################################

            """)

        # Launch omnigibson
        og.launch()

        # Loop over all sample indices to generate individual scenes
        # TODO

        # Load the entire scene
        # scene = VisualizeScene.load_cousin_scene(scene_info=scene_info_path, visual_only=True)
        scene = VisualizeScene.load_cousin_scene(scene_info=scene_info_path, visual_only=False)

        # Make sure all object aren't moving, then step physics once, then resolve collisions
        for obj in scene.objects:
            obj.keep_still()
        og.sim.step_physics()

        all_objects = scene.objects
        print(f"all_objects: {all_objects}")
        print("orientation: (x, y, z, w)")
        for obj in all_objects:
            position, orientation = obj.get_position_orientation()

            print(f"Object Name: {obj.name}")
            # print(f"    Object Name: {obj.name}")
            # print(f"    prim_path: {obj.prim_path}")
            # print(f"    Category: {obj.category}")
            # print(f"    Model: {obj.model}")
            # print(f"    Scale: {obj.scale}")
            # print(f"    UUID: {obj.uuid}")
            # print(f"    - Position: {position}")
            # print(f"    - Orientation: {orientation}")
            obj_info = dict()

            obj_info["root_link"] = {
                "pos": position.tolist(),
                "ori": orientation.tolist(),
                "lin_vel": obj.get_linear_velocity().tolist(),
                "ang_vel": obj.get_angular_velocity().tolist(),
            }

            joint_info = dict()
            if obj.n_joints > 0:
                # print(f"n_joints: {obj.n_joints}")

                obj_joints = obj.joints
                joint_positions = obj.get_joint_positions().tolist()
                joint_velocities = obj.get_joint_velocities().tolist()
                joint_efforts = obj.get_joint_efforts().tolist()
                joint_positions_targets = obj.get_joint_position_targets().tolist()
                joint_velocities_targets = obj.get_joint_velocity_targets().tolist()
                # print(f"obj_joints: {obj_joints}")
                # print(f"joint positions: {joint_positions}")
                # print(f"joint velocities: {joint_velocities}")
                # print(f"joint efforts: {joint_efforts}")
                # print(f"joint positions targets: {joint_positions_targets}")
                # print(f"joint velocities targets: {joint_velocities_targets}")

                for joint_idx, (joint_name, joint_prim) in enumerate(obj_joints.items()):
                    joint_info[joint_name] = {
                        "pos": [joint_positions[joint_idx]],
                        "vel": [joint_velocities[joint_idx]],
                        "effort": [joint_efforts[joint_idx]],
                        "target_pos": [joint_positions_targets[joint_idx]],
                        "target_vel": [joint_velocities_targets[joint_idx]],
                    }
            obj_info["joints"] = joint_info


            obj_info["name"] = obj.name
            obj_info["prim_path"] = obj.prim_path
            obj_info["category"] = obj.category
            obj_info["model"] = obj.model
            obj_info["scale"] = obj.scale.tolist()
            obj_info["uuid"] = obj.uuid


            print(json.dumps(obj_info, indent=4, cls=NumpyTorchEncoder))

        # with open(scene_graph_info_path, "w+") as f:
        #         json.dump(scene_graph_info, f, indent=4, cls=NumpyTorchEncoder)

        for _ in range(10000000):
            og.sim.render()

        print("""

#############################################
### Completed Simulated Scene Generation! ###
#############################################

        """)

        return True

    @staticmethod
    def create_scene(floor=True, sky=True):
        """
        Helper function for creating new empty scene in OmniGibson

        Args:
            floor (bool): Whether to use floor or not
            sky (bool): Whether to use sky or not

        Returns:
            Scene: OmniGibson scene
        """
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    @staticmethod
    def load_cousin_scene(scene_info, visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath

        Args:
            scene_info (dict or str): If dict, scene information to load. Otherwise, should be absolute path to the
                scene info that should be loaded
            visual_only (bool): Whether to load all objects as visual only or not

        Returns:
            Scene: loaded OmniGibson scene
        """
        # Stop sim, clear it, then load empty scene
        scene = VisualizeScene.create_scene(floor=True)

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(f)

        # Set viewer camera to proper pose
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))

        # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                print(f"Object Name: {obj_name}")
                print(f"Category: {obj_info['category']}")
                print(f"Model: {obj_info['model']}")
                print(f"Visual Only: {visual_only}")
                print(f"Scale: {obj_info['scale']}")

                # Object Name: cup_0
                # Category: soda_cup
                # Model: lpanoc
                # Visual Only: True

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

    def take_photo(self, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera

        Args:
            n_render_steps (int): Number of rendering steps to take before taking the photo

        Returns:
            np.ndarray: (H,W,3) RGB frame from viewer camera perspective
        """
        # Render a bit,
        for _ in range(n_render_steps):
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        return rgb


