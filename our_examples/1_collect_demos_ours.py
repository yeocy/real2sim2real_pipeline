"""
Example script for running ACDC to automatically generate simulated scene from single RGB image

Example usage:

python 1_collect_demos.py \
--scene_path ../tests/acdc_output/step_3_output/scene_0/scene_0_info.json \
--target_obj cabinet_4 \
--target_link link_1 \
--cousins bottom_cabinet,bamfsz,link_1 bottom_cabinet_no_top,vdedzt,link_0 \
--dataset_path test_demos.hdf5 \
--n_demos_per_model 3 \
--eval_cousin_id 0 \
--seed 0

"""
from our_method.envs.robomimic.skill_collection_wrapper import SkillCollectionKinovaWrapper
from our_method.envs.robomimic.env_og import EnvOmniGibson

import omnigibson as og
from omnigibson.macros import macros
from omnigibson.utils.constants import JointType
import numpy as np
import torch as th
import argparse
import json

# Hardcode macro values so that open is detected properly
macros.object_states.open_state.JOINT_THRESHOLD_BY_TYPE[JointType.JOINT_REVOLUTE] = 1.0 / 10
macros.object_states.open_state.JOINT_THRESHOLD_BY_TYPE[JointType.JOINT_PRISMATIC] = 1.0 / 3

USE_OSC = False                                                 # Whether to use OSC or not (use IK otherwise)
USE_DELTA_CMDS = False                                           # Whether to use delta commands or not
DIST_USE_FROM_HANDLE = True                                     # Whether to calculate distance from target object handle or center of its bbox
VISUALIZE_SKILL = False                                         # Whether to visualize skill during demo collection or not
XYZ_RANDOMIZATION = np.array([0.03, 0.03, 0.03])                # (x,y,z) randomization to apply to target object
Z_ROT_RANDOMIZATION = np.pi / 30                                # z-rotation randomization to apply to target object                          # z-rotation randomization to apply to target object
EXTERNAL_CAM_XYZ_RANDOMIZATION = np.ones(3) * 0.01              # (x,y,z) randomization to apply to camera between episodes
EXTERNAL_CAM_ROT_RANDOMIZATION = np.pi / 30                     # orientation magnitude randomization to apply to camera between episodes
DEFAULT_DIST_FROM_HANDLE = np.array([0.51, -0.28, -0.265])           # default offset from the robot base to the target object's handle
# DEFAULT_DIST_FROM_HANDLE = np.array([0.5, 0.1, -1.7])           # default offset from the robot base to the target object's handle
BBOX_TRAIN_RANDOMIZATION = np.array([0.1, 0.1, 0.1]) * 1.0   # Relative % randomization scaling of target objects' bboxes
STEP_DIVISOR = 5                                                # Minimum physics steps to take (@60Hz) per action


def main(args):
    # Load cousin scene information
    with open(args.scene_path, "r") as f:
        scene_info = json.load(f)

    # Aggregate cousins
    eval_cousin_id = args.eval_cousin_id
    obj_info = scene_info["objects"][args.target_parent_obj]
    cousin_category_names, cousin_model_names, cousin_link_names = [obj_info["category"]], [obj_info["model"]], [args.target_link]
    if args.cousins is not None:
        for cousin_str in args.cousins:
            cousin_category, cousin_model, cousin_link = cousin_str.split(",")
            cousin_category_names.append(cousin_category)
            cousin_model_names.append(cousin_model)
            cousin_link_names.append(cousin_link)

    n_cousins = len(cousin_category_names)
    randomize_textures = [True] * n_cousins
    randomize_textures[eval_cousin_id] = False
    target_bbox_array = np.array(obj_info["bbox_extent"])
    cab_bboxes = [th.tensor(target_bbox_array, dtype=th.float)] * n_cousins

    # target_bbox_array = np.array(target_bbox)   # adjust distances and bbox_randomization by target_bbox_array
    bbox_train_randomization = BBOX_TRAIN_RANDOMIZATION * target_bbox_array
    bbox_randomizations = np.array([bbox_train_randomization] * n_cousins)
    bbox_randomizations[eval_cousin_id] = 0

    # Set robot params
    robot_params = {
        "model_name": "KinovaGen3Lite",      # Only currently works for FrankaPanda, FrankaMounted models
        "robot_name": "robot0",
        "reset_qpos": th.tensor([0.0, 0.3491, 2.6180, -1.5359, -0.6981, -1.5184, 0.0, 0.0, 0.0, 0.0]),  # 0  20  150  -88  -40  -87          # "reset_qpos": th.tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "eef_z_offset": 0.14,
        # "open_qpos": None,              # If specified, joint values defining an open state for the robot gripper
        "open_qpos": th.tensor([0.96, 0.96, -0.5, -0.5]),              # If specified, joint values defining an open state for the robot gripper
        # "open_qpos": th.tensor([0.95, 0.95, -0.49, -0.49]),              # If specified, joint values defining an open state for the robot gripper # 0, 1: RIGHT_BOTTOM, 2 : Left DIST, 3 : RIGHT_DIST, 
        "closed_qpos": th.tensor([-0.09, -0.09, 0.209, 0.209]),              # If specified, joint values defining an open state for the robot gripper
        "root_link": "BASE",
        # "vis_local_position": th.tensor([-0.3, 0.3, 0.8]),
        "vis_local_position": th.tensor([0.0765683, 0.54672388, 1.01939066]),
        # "vis_local_orientation": th.tensor([-0.30369818, -0.05854882, 0.00282494, 0.95096344]), # X, Y, Z, W
        "vis_local_orientation": th.tensor([-0.05854882, 0.30369818, 0.95096344, -0.00282494]), # X, Y, Z, W
        # "vis_local_orientation": th.tensor([-0.00282494, 0.95096344, -0.30369818, 0.05854882]),
    }

    # # Set robot params
    # robot_params = {
    #     "model_name": "FrankaMounted",      # Only currently works for FrankaPanda, FrankaMounted models
    #     "robot_name": "robot0",
    #     "reset_qpos": th.tensor([-0.0027, -1.3000, -0.0012, -2.0000, -0.0082, 2.1875, 0.8032, 0.0400, 0.0400]),
    #     "eef_z_offset": 0.180,
    #     "open_qpos": None,              # If specified, joint values defining an open state for the robot gripper
    #     "root_link": "panda_base",
    #     "vis_local_position": th.tensor([-0.26549, -0.30288, 1.0 + 0.861]),
    #     "vis_local_orientation": th.tensor([0.36165891, -0.24745751, -0.50752921, 0.74187715]),
    # }

    # Define skill kwargs
    skill_kwargs = dict(
        should_open=True,
        joint_limits=(0.0, np.pi / 4),  # This assumes joint is revolute
        n_approach_steps=int(100 / STEP_DIVISOR),
        n_converge_steps=int(75 / STEP_DIVISOR),
        n_grasp_steps=int(100 / STEP_DIVISOR),
        n_articulate_steps=int(125 / STEP_DIVISOR),
        n_buffer_steps=int(5 / STEP_DIVISOR),
    )

    # Define configuration to load for environment
    physics_frequency = 60
    action_frequency = physics_frequency / STEP_DIVISOR
    output_pos_max = 0.05
    output_rot_max = 0.2
    output_max = np.concatenate([np.ones(3) * output_pos_max, np.ones(3) * output_rot_max])
    controller_output_limits = th.tensor(np.array([-output_max, output_max]), dtype=th.float)

    external_cam_name = "external_cam0"
    cfg = dict()
    
    # Add vis camera
    external_sensors = [
        {
            "sensor_type": "VisionSensor",
            "name": f"{external_cam_name}",
            "relative_prim_path": f"/controllable__{robot_params['model_name'].lower()}__{robot_params['robot_name']}/{robot_params['root_link']}/{external_cam_name}",
            "modalities": ["rgb", "depth_linear", "seg_instance_id"],
            # "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": 480,
                "image_width": 640,
                "focal_length": 20,
                # "intrinsic_matrix" : [[614.04736328125, 0.0, 320.4498596191406], [0.0, 614.044677734375, 244.81724548339844], [0.0, 0.0, 1.0]]
                # "focal_length":  614.044677734375,
            },
            "position": robot_params["vis_local_position"],
            "orientation": robot_params["vis_local_orientation"],
            "pose_frame": "parent",
        }
    ]

    cfg["env"] = {
        "action_frequency": action_frequency,
        "rendering_frequency": action_frequency,
        "physics_frequency": physics_frequency,
        "automatic_reset": False,
        "external_sensors": external_sensors,
    }
    cfg["render"] = {
        "viewer_width": 1080,
        "viewer_height": 1080,
    }
    cfg["scene"] = {
        "type": "Scene",
        "use_floor_plane": True,
    }
    cfg["robots"] = [{
        "type": robot_params["model_name"],
        "name": robot_params["robot_name"],
        "position": np.ones(3) * 100.0,
        "orientation": None,
        "obs_modalities": ["rgb", "depth_linear", "seg_instance_id", "proprio"],
        "scale": 1.0,
        "self_collision": False,
        "action_normalize": True if USE_DELTA_CMDS else False,
        "action_type": "continuous",
        # "grasping_mode": "physical",
        "grasping_mode": "sticky",
        "proprio_obs": ["eef_0_pos", "eef_0_quat", "gripper_0_qpos"],
        "reset_joint_pos": robot_params["reset_qpos"],
        "sensor_config": {
              "VisionSensor": {
                "sensor_kwargs": {
                  "image_height": 128,
                  "image_width": 128,
                },
            },
        },
        "controller_config": {
            "arm_0": {
                "name": "OperationalSpaceController" if USE_OSC else "InverseKinematicsController",
                "mode": "pose_delta_ori" if USE_DELTA_CMDS else "absolute_pose",
                #  "control_limits": {
                #                 "position": [
                #                     [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],  # 최소 위치 (rad)
                #                     [3.14,  3.14,  3.14,  3.14,  3.14,  3.14],   # 최대 위치 (rad)
                #                 ],
                #                 "velocity": [
                #                     [-1.6, -1.6, -1.6, -1.6, -1.6, -3.2],  # 최소 속도 (rad/s)
                #                     [1.6,  1.6,  1.6,  1.6,  1.6,  3.2],   # 최대 속도 (rad/s)
                #                 ],
                #                 "effort": [
                #                     [-10.0, -14.0, -10.0, -7.0, -7.0, -7.0],  # 최소 effort (Nm)
                #                     [10.0,  14.0,  10.0,  7.0,  7.0,  7.0],   # 최대 effort (Nm)
                #                 ],
                #                 "has_limit": [True, True, True, True, True, True],
                #             },
                # "dof_idx": [0, 1, 2, 3, 4, 5],
                "command_input_limits": "default" if USE_DELTA_CMDS else None,
                "command_output_limits": controller_output_limits if USE_DELTA_CMDS else None,


                # "kp": 200.0,  # 비례게인 (stiffness), 필요 시 조정
                # "damping_ratio": 1.0,  # critical damping
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                # "mode": "binary",
                 "control_limits": {
                                "Position": [
                                    [-2.7600, -2.7600, -2.7600, -2.6700, -2.6700, -2.6700, -0.09, -0.09, -0.209, -0.209],  # 최소 위치 (rad)
                                    [2.7600, 2.7600, 2.7600, 2.6700, 2.6700, 2.6700, 0.9600, 0.9600,  0.49, 0.49],   # 최대 위치 (rad)
                                ],
                                # "velocity": [
                                #     [-1.6, -1.6, -1.6, -1.6, -1.6, -3.2],  # 최소 속도 (rad/s)
                                #     [1.6,  1.6,  1.6,  1.6,  1.6,  3.2],   # 최대 속도 (rad/s)
                                # ],
                                # "effort": [
                                #     [-10.0, -14.0, -10.0, -7.0, -7.0, -7.0],  # 최소 effort (Nm)
                                #     [10.0,  14.0,  10.0,  7.0,  7.0,  7.0],   # 최대 effort (Nm)
                                # ],
                                # "has_limit": [True, True, True, True, True, True],
                            },
                "open_qpos": robot_params["open_qpos"],
                "closed_qpos": robot_params["closed_qpos"],
            },
        },
    }]
# "open_qpos": th.tensor([0.96, 0.96, -0.5, -0.5]),              # If specified, joint values defining an open state for the robot gripper
#         # "open_qpos": th.tensor([0.95, 0.95, -0.49, -0.49]),              # If specified, joint values defining an open state for the robot gripper # 0, 1: RIGHT_BOTTOM, 2 : Left DIST, 3 : RIGHT_DIST, 
#         "closed_qpos": th.tensor([-0.09, -0.09, 0.209, 0.209]),              # If specified, joint values defining an open state for the robot gripper
        
    # AssertionError: Invalid wrapper type received! 
    # Valid options are: dict_keys(['DataWrapper', 'DataCollectionWrapper', 'DataPlaybackWrapper', 'SkillWrapper', 'OpenCabinetWrapper', 'SkillCollectionWrapper']), got: CabinetPickAndPlaceWrapper

    cfg["wrapper"] = {
        # "type": "OpenCabinetWrapper",
        # "type": "PickCupInTheCabinetWrapper",
        "type": "PickCupInTheCabinetKinovaWrapper",
        "eef_z_offset": robot_params["eef_z_offset"],
        "cab_categories": cousin_category_names,
        "cab_models": cousin_model_names,
        "cab_links": cousin_link_names,
        "cab_bboxs": cab_bboxes,
        "eval_idx": eval_cousin_id,
        "handle_dist": 0.005,
        "dist_use_from_handle": DIST_USE_FROM_HANDLE,
        "dist_out_from_handle": DEFAULT_DIST_FROM_HANDLE[0],
        "dist_right_of_handle": DEFAULT_DIST_FROM_HANDLE[1],
        "dist_up_from_handle": DEFAULT_DIST_FROM_HANDLE[2],
        # "z_rot_from_handle": -th.pi / 2,
        "xyz_randomization": XYZ_RANDOMIZATION,
        "z_rot_randomization": Z_ROT_RANDOMIZATION,
        "bbox_randomization": bbox_randomizations,
        "randomize_textures": randomize_textures,
        "randomize_agent_pose": True,
        "randomize_cabinet_pose": False,
        "skill_kwargs": skill_kwargs,
        "use_delta_commands": USE_DELTA_CMDS,
        "visualize_cam_pose": (
            robot_params["vis_local_position"],
            robot_params["vis_local_orientation"],
        ),
        "visualize_skill": VISUALIZE_SKILL,
        "custom_bddl": None,
        "task_activity_name": "open_cabinet",
        "scene_info": scene_info,
        "scene_target_parent_obj_name": args.target_parent_obj,
        "scene_target_child_obj_name": args.target_child_obj,
    }

    # Create the robomimic-compatible environment
    # ACDC Scene 생성
    env = EnvOmniGibson(
        env_name="test_env",
        obs_modalities=[
            # f"external::{external_cam_name}::rgb",                # Can include RGB, depth if requested
            # f"external::{external_cam_name}::depth_linear",
            f"external::{external_cam_name}::point_cloud",
            f"{robot_params['robot_name']}::proprio",
        ],
        combine_pc=True,
        include_segment_strs=["cabinet","pencil_case"],
        include_eef_pc=True,
        embed_eef_pc=True,
        max_pc=2048,
        og_config=cfg,
        postprocess_visual_obs=False,
        render=True,
        wrap_during_initialization=True,
        robot_cam_depth_threshold=10,
        external_cam_depth_threshold=100,
        external_cam_xyz_randomization=EXTERNAL_CAM_XYZ_RANDOMIZATION,
        external_cam_rot_randomization=EXTERNAL_CAM_ROT_RANDOMIZATION,
        prune_depth_background=True,
        n_loops_until_setpoint_reached=10,
        init_robot_joint_noise_proportion=0.02,
    )

    og.sim.viewer_camera.set_position_orientation(*scene_info["cam_pose"])
    
    # # Wrap with skill data collection
    # env = SkillCollectionWrapper(
    #     env=env,
    #     path=args.dataset_path,
    #     only_successes=True,
    #     use_delta_commands=USE_DELTA_CMDS,
    # )
    # Wrap with skill data collection
    env = SkillCollectionKinovaWrapper(
        env=env,
        path=args.dataset_path,
        only_successes=True,
        use_delta_commands=USE_DELTA_CMDS,
    )

    # Execute the skill
    video_count = 0
    buffer_traj_count = 0

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    for i, model in enumerate(cousin_model_names):
        if i == eval_cousin_id:
            # Skip this one, since it's the eval model
            continue
        n_cab_successes = 0
        while n_cab_successes < args.n_demos_per_model:
            # Update the cabinet, reset, collect demo
            env.env.env.set_cabinet_idx(idx=i)
            env.reset()
            env.collect_demo()
            if env.env.is_success()["task"]:
                n_cab_successes += 1
                buffer_traj_count += 1
                env.flush_current_traj()
                og.log.warning(f"Collected model {model} successful demo n: {n_cab_successes}")
            video_count += 1

            if buffer_traj_count >= 50:
                env.flush_current_file()    # flush to avoid large memory footprint
                buffer_traj_count = 0

    # Save data and close
    env.save_data()
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True,
                        help="Absolute path to input cousin scene .json file to use")
    parser.add_argument("--target_parent_obj", type=str, required=True,
                        help="Name of the object to articulate")
    parser.add_argument("--target_child_obj", type=str, required=True,
                        help="Name of the object to articulate")
    parser.add_argument("--target_link", type=str, required=True,
                        help="Name of @target_obj's link to articulate")
    parser.add_argument("--cousins", type=str, nargs="+", default=None,
                        help="If specified, <category>,<model>,<link> string(s) representing cousins to additionally swap out for @target_obj when collecting demos and / or evaluating a trained policy")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Absolute path to output dataset file to generate")
    parser.add_argument("--n_demos_per_model", type=int, default=5,
                        help="Number of demos to collect per model")
    parser.add_argument("--eval_cousin_id", type=int, default=0,
                        help="Cousin ID to hold out from training for zero-shot eval later")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed to use for randomization")
    args = parser.parse_args()
    main(args)
