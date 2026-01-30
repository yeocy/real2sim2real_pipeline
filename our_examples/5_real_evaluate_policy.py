"""
Example usage:

python 4_evaluate_policy.py \
--agent ../training_results/default_run/20241004141334/models/model_epoch_4.pth

# This run is for reproducing paper results
python 4_evaluate_policy.py \
--agent ../training_results/twin_ckpt.pth \
--eval_category_model_link_name bottom_cabinet,dajebq,link_3 \
--n_rollouts 100 \
--seed 1

# To change aggressiveness of randomization during evaluation, you can pass the following optional argument:
--eval_bbox_rand 0.25,0.25,0.25
--eval_xyz_rand 0.03,0.03,0.07
--eval_z_rot_rand 0.314
"""


# Necessary to make sure robomimic registers these modules
from grpc import UnaryUnaryMultiCallable
from robomimic import algo
# import digital_cousins

import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import actionlib
import rospy
import moveit_commander
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import torch
import open3d as o3d
import fpsample
import tf.transformations as tf_trans
import torch as th

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# 현재 스크립트 기준, our_method_test/utils 경로를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 상위 디렉토리 (real2sim2real_pipeline)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

UTILS_DIR = os.path.join(PROJECT_ROOT, "our_method/utils")
sys.path.append(UTILS_DIR)

from robomimic_utils import *

from robomimic.utils import tensor_utils as TensorUtils
from robomimic.models.obs_core import Randomizer, ColorRandomizer, CropRandomizer, GaussianNoiseRandomizer, EncoderCore
from robomimic.utils.vis_utils import visualize_image_randomizer
import robomimic.models.base_nets as BaseNets
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
from math import pi, radians
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler
from message_filters import Subscriber, TimeSynchronizer
from message_filters import Subscriber, ApproximateTimeSynchronizer

# import omnigibson as og
NAMESPACE = "/my_gen3_lite" # Set to your namespace if using one (e.g., "/gen3_lite")
import cv2
from cv_bridge import CvBridge, CvBridgeError

# # Modify default cameras
# DEFAULT_CAMERAS[EB.EnvType.OMNIGIBSON_TYPE] = [None]    # None corresponds to viewer camera

class EnvRealKinova():
    """Wrapper class for real panda environment"""
    def __init__(
        self,
        namespace_primitive="/my_gen3_lite",
        cam_rgb_topic = "/camera/color/image_raw",
        cam_point_topic = "/camera/depth/color/points",
        cam_depth_topic = "/camera/aligned_depth_to_color/image_raw",
        move_group_arm="arm",
        move_group_gripper="gripper",
        end_effector_link = "tool_frame"
    ):
        """
        Args:
            env_name (str): name of environment.

            render (bool): ignored - on-screen rendering is not supported

            render_offscreen (bool): ignored - image observations are supplied by default

            use_image_obs (bool): ignored - image observations are used by default.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            control_freq (int): real-world control frequency to try and enforce through rate-limiting

            action_scale (list): list of 7 numbers for what the -1 and 1 action in each dimension corresponds to
                for the physical robot action space

            camera_names_to_sizes (dict):  dictionary that maps camera names to tuple of image height and width
                to return
        """
        # assert (action_scale is not None), "must provide action scaling bounds"
        # assert len(action_scale) == 7, "must provide scaling for all dimensions"
        self.namespace = rospy.get_param('~namespace', namespace_primitive)

        self.count = 0
        self.rgb_count = 0
        self.depth_count = 0

        self.robot = None
        self.scene = None
        self.arm_group = None
        self.end_effector_link = end_effector_link

        self.is_init_success = False
        self.rgb_image = None
        self.depth_image = None

        self.bridge = CvBridge()
        # self.rgb_image_sub = rospy.Subscriber(cam_rgb_topic, Image, self.rgb_image_callback)
        # self.depth_image_sub = rospy.Subscriber(cam_depth_topic, Image, self.depth_image_callback)

        # RGB, Depth subscriber
        self.rgb_sub = Subscriber(cam_rgb_topic, Image)
        self.depth_sub = Subscriber(cam_depth_topic, Image)

        # 정확한 시간 동기화: queue_size=10, slop=0.05s
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=50, slop=0.15)
        self.ts.registerCallback(self.synced_rgbd_callback)

        self.fx = 905.757568359375
        self.fy = 906.0278930664062
        self.cx = 659.2559814453125
        self.cy = 361.3847961425781
        self.width = 1280
        self.height = 720
        self.camera_calibration = [[-0.99991957,  0.00719842, -0.01044224,  0.00826274],
                                [ 0.01246631,  0.70935772, -0.7047384,   1.0580069 ],
                                [ 0.00233428, -0.70481189, -0.70939041,  0.76899746],
                                [ 0.,          0.,          0.,          1.,        ]]

        # Open3D용 Intrinsic 객체 생성
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(self.width, self.height, self.fx, self.fy, self.cx, self.cy)



        # self.point_sub = rospy.Subscriber(cam_point_topic, PointCloud2, self.point_callback)

        rospy.loginfo("Initializing KinovaRobot...")

        try:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('kinova_controller', anonymous=True)


            
            
            # self.rgb_sub = Subscriber(cam_rgb_topic, Image)
            # self.depth_sub = Subscriber(cam_depth_topic, Image)

            # # 정확한 타임스탬프 동기화 (시간이 완전히 같아야 함)
            # self.ts = TimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10)
            # self.ts.registerCallback(self.image_callback)

            full_namespace = self.namespace if self.namespace.startswith('/') else '/' + self.namespace
            if full_namespace == '/': full_namespace = "" # Handle empty namespace correctly

            robot_description = full_namespace + "/robot_description"
            rospy.loginfo(f"Waiting for robot_description parameter at: {robot_description}")
            # Add a wait for the parameter to be available
            start_time = rospy.Time.now()
            while not rospy.has_param(robot_description) and (rospy.Time.now() - start_time).to_sec() < 10.0:
                rospy.sleep(0.5)
            if not rospy.has_param(robot_description):
                 raise rospy.ROSException(f"Parameter {robot_description} not found after waiting.")


            self.robot = moveit_commander.RobotCommander(robot_description=robot_description, ns=full_namespace)
            self.scene = moveit_commander.PlanningSceneInterface(ns=full_namespace)
            self.arm_group = moveit_commander.MoveGroupCommander(move_group_arm, robot_description=robot_description, ns=full_namespace)
            self.gripper_group = moveit_commander.MoveGroupCommander(move_group_gripper, robot_description=robot_description, ns=full_namespace)

            # Check if end-effector link exists
            if self.end_effector_link not in self.robot.get_link_names():
                 rospy.logwarn(f"End effector link '{self.end_effector_link}' not found in robot model!")
                 rospy.logwarn(f"Available links: {self.robot.get_link_names()}")
                 # You might want to raise an error or use a default link if appropriate
                 # raise ValueError(f"End effector link '{END_EFFECTOR_LINK}' not found!")
            self.arm_group.set_end_effector_link(self.end_effector_link)
            self.arm_group.set_planning_time(10.0) # Allow more time for planning complex poses MoveIt이 경로를 찾는 데 최대 10초까지 시도
            self.arm_group.set_goal_position_tolerance(0.01) # meters  1cm 이내면 도달한 것으로 인정
            self.arm_group.set_goal_orientation_tolerance(0.05) # radians 라디안 기준 ±0.05 이내

            # Ensure DOF is correctly detected (useful for joint targets)
            self.degrees_of_freedom = len(self.arm_group.get_active_joints())
            self.gripper_degrees_of_freedom = len(self.arm_group.get_active_joints())
            rospy.loginfo(f"Detected {self.degrees_of_freedom} DOF for group '{move_group_arm}'.")
            rospy.loginfo(f"Detected {self.gripper_degrees_of_freedom} DOF for group '{move_group_gripper}'.")


            self.gripper_client = actionlib.SimpleActionClient(
                "/my_gen3_lite/gen3_lite_2f_gripper_controller/gripper_cmd",
                GripperCommandAction
            )
            rospy.loginfo("Waiting for gripper action server...")
            self.gripper_client.wait_for_server()
            rospy.loginfo("Gripper action server connected!")


            rospy.loginfo("MoveIt objects initialized successfully.")
            self.is_init_success = True

            self.go_init()
            self.close_gripper()
            rospy.sleep(2.0)
            # self.open_gripper()

        

        except Exception as e:
            rospy.logerr(f"Failed to initialize MoveIt objects: {e}")
            import traceback
            traceback.print_exc()
            self.is_init_success = False

    def step(self, action, n_steps=1):
        """
        Executes a 7D delta action + gripper command on the real Kinova robot.

        Args:
            action (np.array): (7,) array. 
                - First 6 entries: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)
                - Last entry: gripper command (-1 for close, 1 for open)
            n_steps (int): How many control steps to wait after sending the command (default 1)

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """
        if not self.is_init_success:
            raise RuntimeError("Robot is not initialized properly!")

        assert len(action) == 7, "Action must be 7D: (dx, dy, dz, droll, dpitch, dyaw, gripper)"
        gripper_cmd = action[6]
        action = action * 0.05

        # 1. 액션 분리
        delta_pos = action[:3]
        delta_aa = action[3:6]
        

        # 2. 현재 pose 읽기
        pose_stamped = self.get_current_pose_stamped()
        curr_pos = np.array([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z
        ])
        curr_quat = np.array([
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w
        ])

        # 3. 목표 pose 계산
        target_pos = curr_pos + delta_pos
        delta_quat = self.axisangle2quat(delta_aa)   # delta axis-angle → quaternion
        target_quat = self.quat_multiply(delta_quat, curr_quat)  # 현재 orientation에 delta 회전 적용

        target_rpy = list(tf_trans.euler_from_quaternion(target_quat))
        # print(f"target_rpy: {target_rpy}")


        print(gripper_cmd)
        # # 4. Gripper 명령 적용
        if gripper_cmd <= -0.0:
            self.close_gripper()
        elif gripper_cmd >= 0.0:
            self.open_gripper()
        rospy.sleep(0.3)
        # print(f"current_pos: {curr_pos}")
        # print(f"delta_pos: {delta_pos}")
        # print(f"target_pos: {target_pos}")
        # print(f"current_rpy (degree): {np.degrees(tf_trans.euler_from_quaternion(curr_quat))}")

        # print(f"delta_rpy (degree): {delta_aa}")
        # print(f"target_rpy (degree): {np.degrees(tf_trans.euler_from_quaternion(target_quat))}")
        # 4. IK로 이동

        move_success = self.move_to_pose_target(
            position=target_pos.tolist(),
            orientation_rpy=target_rpy
        )


        if not move_success:
            print("[❌] Move to target pose failed!")

            

        # (중간 값이면 무시)

        # 5. 약간 대기
        rospy.sleep(0.05 * n_steps)

        # 6. 관측값 가져오기
        obs = self.get_observation()

        # 7. reward, terminated, truncated, info 설정
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def axisangle2quat(self, vec, eps=1e-6):
        """
        Converts scaled axis-angle vector to quaternion using numpy.

        Args:
            vec (np.ndarray): (..., 3) axis-angle exponential coordinates
            eps (float): stability threshold

        Returns:
            np.ndarray: (..., 4) quaternion (x, y, z, w)
        """
        vec = np.asarray(vec)

        # (3,) → (1,3)으로 확장
        if vec.ndim == 1:
            if vec.shape[0] != 3:
                raise ValueError(f"Expected vec with shape (3,), got {vec.shape}")
            vec = vec[None, :]  # (3,) -> (1,3)

        input_shape = vec.shape[:-1]
        flat_vec = vec.reshape(-1, 3)

        angle = np.linalg.norm(flat_vec, axis=-1, keepdims=True)

        quat = np.zeros((flat_vec.shape[0], 4), dtype=np.float32)
        quat[:, 3] = 1.0  # 기본적으로 w=1 (identity)

        idx = (angle.reshape(-1) > eps)
        if np.any(idx):
            sin_half_angle = np.sin(angle[idx] * 0.5)
            quat[idx, :3] = flat_vec[idx] * (sin_half_angle / angle[idx])
            quat[idx, 3] = np.cos(angle[idx] * 0.5).reshape(-1)

        quat = quat.reshape(*input_shape, 4)

        # 🎯 싱글 input이면 (4,)로 squeeze!
        if quat.shape[0] == 1:
            quat = quat.reshape(4)

        return quat


    def quat_multiply(self, q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """
        Return multiplication of two quaternions (q1 * q0) using numpy.

        Args:
            q1 (np.ndarray): (4,) or (..., 4) array, (x,y,z,w) quaternion
            q0 (np.ndarray): (4,) or (..., 4) array, (x,y,z,w) quaternion

        Returns:
            np.ndarray: (4,) or (..., 4) array, (x,y,z,w) multiplied quaternion
        """
        q1 = np.asarray(q1)
        q0 = np.asarray(q0)

        x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]

        result = np.stack([
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ], axis=-1)

        return result


    def get_current_pose_stamped(self):
        """Gets the current pose of the end-effector link."""
        if not self.is_init_success: return None
        try:
            # Ensure we get the pose relative to the base frame (usually robot's root link)
            base_frame = self.robot.get_planning_frame()
            pose_stamped = self.arm_group.get_current_pose(self.end_effector_link)
            # Double-check the frame_id, transform if necessary (usually MoveIt handles this)
            if pose_stamped.header.frame_id != base_frame and pose_stamped.header.frame_id.lstrip('/') != base_frame.lstrip('/'):
                 rospy.logwarn_throttle(10, f"Pose frame '{pose_stamped.header.frame_id}' differs from planning frame '{base_frame}'. Ensure TF is correct.")
                 # Ideally, transform the pose here if needed, but MoveIt often returns it in base frame
            return pose_stamped
        except Exception as e:
            rospy.logerr(f"Failed to get current pose for link '{self.end_effector_link}': {e}")
            return None
        
    def define_target_poses(self, num_poses):
        """
        Defines a list of target joint configurations.
        IMPORTANT: Adjust these poses for your specific robot (Gen3 Lite 6DOF)
                   and camera setup to ensure pattern visibility!
        These are just examples, likely need significant tuning.
        Angles are in RADIANS.
        """
        target_poses = []

        # --- Example Poses for 6 DOF (Gen3 Lite) ---
        # Ensure these poses provide diverse views of the pattern for the camera
        # Start near 'home' or a known good viewing pose and add variations.

        # Pose 1: Near Home, slightly tilted

        target_poses.append([radians(-23), radians(-20), radians(40), radians(-25), radians(-111), radians(2)])
        target_poses.append([radians(-30), radians(8), radians(75), radians(-32), radians(-130), radians(-20)])

        # target_poses.append([radians(0), radians(15), radians(90), radians(0), radians(-45), radians(0)])
        # target_poses.append([radians(-142), radians(79), radians(113), radians(98), radians(-114), radians(84)])
        # target_poses.append([radians(-93), radians(22), radians(117), radians(10), radians(-139), radians(-6)])
        # target_poses.append([radians(51), radians(88), radians(129), radians(-97), radians(-38), radians(85)])
        # target_poses.append([radians(51), radians(88), radians(132), radians(-112), radians(-38), radians(96)])
        # target_poses.append([radians(41), radians(81), radians(105), radians(-95), radians(-54), radians(83)])
        # target_poses.append([radians(-20), radians(22), radians(40), radians(121), radians(127), radians(-143)])
        # target_poses.append([radians(23), radians(20), radians(50), radians(123), radians(90), radians(-130)])
        # target_poses.append([radians(26), radians(22), radians(55), radians(119), radians(95), radians(-124)])
        # target_poses.append([radians(26), radians(22), radians(55), radians(91), radians(95), radians(-138)])
        # target_poses.append([radians(32), radians(13), radians(52), radians(97), radians(97), radians(-138)])
        # target_poses.append([radians(8), radians(1), radians(24), radians(-64), radians(-116), radians(34)])
        # target_poses.append([radians(8), radians(1), radians(24), radians(-60), radians(-116), radians(34)])
        # target_poses.append([radians(8), radians(-26), radians(24), radians(-60), radians(-116), radians(34)])
        # target_poses.append([radians(10), radians(-21), radians(70), radians(-48), radians(-100), radians(8)])

        # target_poses.append([radians(17), radians(-10), radians(66), radians(-56), radians(-96), radians(5)])
        # target_poses.append([radians(-5), radians(-22), radians(75), radians(-19), radians(-105), radians(-19)])
        # target_poses.append([radians(-6), radians(-18), radians(56), radians(-33), radians(-113), radians(-5)])
        # target_poses.append([radians(-13), radians(-23), radians(43), radians(-28), radians(-122), radians(0)])
        # target_poses.append([radians(-33), radians(4), radians(81), radians(-37), radians(-133), radians(-24)])
        # --- End Example Poses ---

        if len(target_poses) < num_poses:
            rospy.logwarn(f"Requested {num_poses} poses, but only {len(target_poses)} defined. Using available poses.")
            return target_poses
        else:
            # If more poses defined than needed, truncate the list
            return target_poses[:num_poses]


    def move_to_joint_target(self, joint_target_rad):
        """Moves the arm to a specific joint configuration."""
        if not self.is_init_success: return False
        if len(joint_target_rad) != self.degrees_of_freedom:
             rospy.logerr(f"Incorrect number of joint angles provided. Expected {self.degrees_of_freedom}, got {len(joint_target_rad)}")
             return False

        rospy.loginfo(f"Planning and moving to joint target: {[f'{q:.2f}' for q in joint_target_rad]}")
        self.arm_group.set_joint_value_target(joint_target_rad)
        success = self.arm_group.go(wait=True)

        self.arm_group.stop() # Ensure robot stops
        self.arm_group.clear_pose_targets() # Clear targets

        if not success:
            rospy.logerr("Failed to reach joint target.")
        else:
            rospy.loginfo("Reached joint target successfully.")
            
        return success

    def move_to_pose_target(self, position, orientation_rpy):
        """ 
        Move to a desired pose using IK.

        Args:
            position (list): [x, y, z]
            orientation_rpy (list): [roll, pitch, yaw] in radians
        """
        if not self.is_init_success:
            rospy.logerr("MoveIt not initialized.")
            return False

        pose_target = Pose()
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]

        q = quaternion_from_euler(*orientation_rpy)
        pose_target.orientation.x = q[0]
        pose_target.orientation.y = q[1]
        pose_target.orientation.z = q[2]
        pose_target.orientation.w = q[3]

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.robot.get_planning_frame()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose_target

        self.arm_group.set_pose_target(pose_stamped)
        # set_pose_target : 현재 위치에서 Target Pose
        success = self.arm_group.go(wait=True)

        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if not success:
            rospy.logerr("Failed to move to desired pose.")
        else:
            rospy.loginfo("Successfully moved to desired pose.")
        return success

    def go_retract(self):
        """Moves the robot to the 'home' named target."""
        if not self.is_init_success: return False
        rospy.loginfo("Moving to 'retract' position...")
        self.arm_group.set_named_target("retract")
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        if not success:
            rospy.logerr("Failed to reach 'retract' position.")
        else:
            rospy.loginfo("Reached 'retract' position.")
        return success

    def go_home(self):
        """Moves the robot to the 'home' named target."""
        if not self.is_init_success: return False
        rospy.loginfo("Moving to 'home' position...")
        self.arm_group.set_named_target("home")
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        if not success:
            rospy.logerr("Failed to reach 'home' position.")
        else:
            rospy.loginfo("Reached 'home' position.")
        return success
    
    def go_init(self):
        """Moves the robot to the given initial joint position."""
        if not self.is_init_success:
            return False

        rospy.loginfo("Moving to 'init' joint position...")
        
        # 원하는 joint angles (rad 단위)
        init_joint_positions = [1.5708, 0.3491, 2.6, -1.5359, -0.6981, -1.5184]

        # moveit group에 설정
        self.arm_group.set_joint_value_target(init_joint_positions)
        
        # 이동 명령
        success = self.arm_group.go(wait=True)
        
        # 멈추고, 타겟 클리어
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if not success:
            rospy.logerr("Failed to reach 'init' joint position.")
        else:
            rospy.loginfo("Reached 'init' joint position.")
        
        return success
    
    def close_gripper(self):
        """Closes the Kinova 2F gripper via GripperCommand Action."""
        if not self.is_init_success:
            rospy.logerr("MoveIt not initialized. Cannot close gripper.")
            return False

        try:
            goal = GripperCommandGoal()
            goal.command.position = 0.0      # 0.0 = 완전히 닫힘
            # goal.command.max_effort = 50.0    # 충분한 힘 설정

            rospy.loginfo("Sending gripper close goal...")
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result()

            result = self.gripper_client.get_result()
            rospy.loginfo(f"Gripper close result: {result}")
            return True

        except Exception as e:
            rospy.logerr(f"Failed to close gripper: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def open_gripper(self):
        """Opens the Kinova 2F gripper via GripperCommand Action."""
        if not self.is_init_success:
            rospy.logerr("MoveIt not initialized. Cannot open gripper.")
            return False

        try:
            goal = GripperCommandGoal()
            goal.command.position = 0.9    # 0.8 = 충분히 열림 (0.9까지 열 수도 있음)
            goal.command.max_effort = 50.0  # 충분한 힘 설정

            rospy.loginfo("Sending gripper open goal...")
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result()

            result = self.gripper_client.get_result()
            rospy.loginfo(f"Gripper open result: {result}")
            return True

        except Exception as e:
            rospy.logerr(f"Failed to open gripper: {e}")
            import traceback
            traceback.print_exc()
            return False


    def run_collection(self, num_poses_to_collect,):
        """Executes the data collection process."""
        if not self.is_init_success:
            rospy.logerr("Initialization failed. Cannot run collection.")
            return
        current_pose_stamped = self.get_current_pose_stamped()

        target_joint_poses = self.define_target_poses(num_poses_to_collect)
        if not target_joint_poses:
            rospy.logerr("No target poses defined. Exiting.")
            return

        rospy.loginfo(f"Starting data collection for {len(target_joint_poses)} poses.")

        # Go home first
        if not self.go_retract():
            rospy.logerr("Failed to reach home position initially. Aborting.")
            return
        rospy.sleep(2.0) # Short pause after reaching home
        self.get_observation()
        for i, target_joints in enumerate(target_joint_poses):
            pose_index = i + 1

            rospy.loginfo(f"\n--- Moving to Pose {pose_index}/{len(target_joint_poses)} ---")

            # if not self.move_to_joint_target(target_joints):
            #     rospy.logwarn(f"Skipping pose {pose_index} due to movement failure.")
            #     # Ask user if they want to continue
            #     try:
            #          cont = input("Movement failed. Continue to next pose? (y/n): ").lower()
            #          if cont != 'y':
            #               rospy.loginfo("Aborting collection.")
            #               break
            #     except EOFError: # Handle case where input is piped or unavailable
            #          rospy.logerr("Input stream closed. Aborting collection.")
            #          break
            #     continue # Skip to the next pose if user agrees
            # [0.2, 0, 0.2], orientation_rpy=[0, 180, 0]

            if not self.move_to_pose_target(position = [0.3, -0.1, 0.2], orientation_rpy=[0, 3.14159, 1.5708]):
                rospy.logwarn(f"Skipping pose {pose_index} due to movement failure.")
                # Ask user if they want to continue
                try:
                     cont = input("Movement failed. Continue to next pose? (y/n): ").lower()
                     if cont != 'y':
                          rospy.loginfo("Aborting collection.")
                          break
                except EOFError: # Handle case where input is piped or unavailable
                     rospy.logerr("Input stream closed. Aborting collection.")
                     break
                continue # Skip to the next pose if user agrees
            
            rospy.loginfo(f"\n--- Move Success {pose_index}/{len(target_joint_poses)} ---")
            rospy.sleep(2) # Wait for robot to settle
            self.get_observation()

    def synced_rgbd_callback(self, rgb_msg, depth_msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        # print("✅ Updated RGB and Depth images")

        # print("##############################################3")
        rgb_save_dir = "save_rgb"
        os.makedirs(rgb_save_dir, exist_ok=True)

        rgb_save_path = os.path.join(rgb_save_dir, f"{self.count}.png")

        ################################################################################
        depth_save_dir = "save_depth"
        os.makedirs(depth_save_dir, exist_ok=True)

        depth_save_path = os.path.join(depth_save_dir, f"{self.count}.png")

        # 둘 다 같은 시점의 이미지
        # print("✅ Synced pair:", rgb_msg.header.stamp, depth_msg.header.stamp)

        # 예: 저장
        cv2.imwrite(rgb_save_path, self.rgb_image)
        cv2.imwrite(depth_save_path, self.depth_image)

        self.count += 1

    # def image_callback(self, rgb_msg, depth_msg):
    def rgb_image_callback(self, rgb_msg):
        try:
            # Convert ROS Image messages to OpenCV format
            print("RGB Image: ", rgb_msg.header.stamp)
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")       # 컬러
            # depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")  # 뎁스 (16비트 1채널)

            # 저장 디렉토리 만들기
            save_dir = "save_rgb"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{self.rgb_count}.png")
            # Save
            cv2.imwrite(save_path, self.rgb_image)
            rospy.loginfo(f"✅ Saved RGB Image to {save_path}")
            
            self.rgb_count += 1

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

    def depth_image_callback(self, depth_msg):
        try:
            # Convert ROS Image messages to OpenCV format (16-bit 1-channel)
            print("Depth Image:", depth_msg.header.stamp)
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 저장 디렉토리 만들기
            save_dir = "save_depth"
            os.makedirs(save_dir, exist_ok=True)

            # 저장 경로 (.png 또는 .npz 선택)
            save_path = os.path.join(save_dir, f"{self.depth_count}.png")

            # 저장: 16-bit PNG로 저장 (OpenCV는 depth 이미지도 저장 가능)
            cv2.imwrite(save_path, self.depth_image)
            rospy.loginfo(f"✅ Saved Depth Image to {save_path}")

            self.depth_count += 1

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

    def get_observation(self, obs=None):
        # 이미지 들어올 때까지 기다림
        wait_start = rospy.Time.now()
        timeout = 5.0  # 최대 5초 기다리기

        while (self.rgb_image is None or self.depth_image is None) and (rospy.Time.now() - wait_start).to_sec() < timeout:
            rospy.logwarn_throttle(1.0, "Waiting for rgb_image and depth_image to be available...")
            rospy.sleep(0.1)

        # 만약 timeout이 지나도 None이면 에러 발생
        if self.rgb_image is None or self.depth_image is None:
            raise RuntimeError("Failed to receive rgb_image or depth_image within timeout!")
        # self.timers.tic("get_observation")
        # observation = {}
        # observation["ee_pose"] = np.concatenate(self.robot_interface.ee_pose)
        # observation["joint_positions"] = self.robot_interface.joint_position
        # observation["joint_velocities"] = self.robot_interface.joint_velocity
        # observation["gripper_position"] = self.robot_interface.gripper_position
        # observation["gripper_velocity"] = self.robot_interface.gripper_velocity
        # for cam_name in self.camera_names_to_sizes:
        #     im = self.robot_interface.get_camera_frame(camera_name=cam_name)
        #     if self.postprocess_visual_obs:
        #         im = ObsUtils.process_image(im)
        #     observation[cam_name] = im
        # self.timers.toc("get_observation")
        pose_stamped = self.arm_group.get_current_pose(self.end_effector_link)
        gripper_joint_values = self.gripper_group.get_current_joint_values()
        pose = pose_stamped.pose
        eef_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        eef_quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        robot_priop = np.concatenate([eef_pos, eef_quat, gripper_joint_values])  # shape: (7,)

        # BGR to RGB
        color_img = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)

        # depth: mm -> meters
        depth_img = self.depth_image.astype(np.float32) / 1000.0

        # Open3D 이미지 객체 생성
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)

        # RGBD 이미지로 결합
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=0.9,
            depth_trunc=2  # 최대 depth 거리 (3m 이후는 자름)
        )

        # Point cloud 생성
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsics
        )

        pcd.transform(self.camera_calibration)

        # ✅ 여기 추가
        translation_vector = np.array([0.0, 0.1, 0.1])  # 원하는 이동
        pcd.translate(translation_vector)

        # (6) Statistical Outlier 제거 (Noise point 제거)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)



        # (7) numpy로 포인트 변환
        xyz_world = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        x_outside_mask = (xyz_world[:, 0] < -0.3) | (xyz_world[:, 0] > 0.37)
        y_outside_mask = (xyz_world[:, 1] < -0.2) | (xyz_world[:, 1] > 0.75)

        # x, y, z 중 하나라도 범위 밖이면 True
        outside_mask = x_outside_mask | y_outside_mask

        xyz_world = xyz_world[~outside_mask]

        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(xyz_world[:, :3], 2048, h=7)
        xyz_world = xyz_world[kdline_fps_samples_idx]

        # (2048, 4)로 만들기
        zeros = np.zeros((xyz_world.shape[0], 1), dtype=np.float32)
        xyz_world = np.hstack((xyz_world.astype(np.float32), zeros))

        # 저장 경로 (.png 또는 .npz 선택)
        save_path = os.path.join("./point_cloud", f"{self.count}.npz")
        # ✅ npz로 저장
        np.savez(save_path, points=xyz_world, colors=colors)

            # 최종 관절값
        robot_priop = robot_priop.astype(np.float32)
        last4 = robot_priop[-4:]
        reordered = [-1*last4[0], last4[2], -1*last4[1], -1*last4[3]]
        robot_priop[-4:] = reordered

        print(f"robot_priop: {robot_priop}")

        obs = {
            'robot0::proprio': robot_priop,
            'combined::point_cloud': xyz_world
        }

        # # 색상도 함께 필터링
        # colors = np.asarray(pcd.colors)
        # filtered_colors = colors[keep_mask]

        # # 새로운 포인트 클라우드 생성
        # filtered_pcd = o3d.geometry.PointCloud()
        # filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        # filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)


        joint_values = self.arm_group.get_current_joint_values()

        # print(obs)
        return obs

def point_cloud_test(policy):
    data = np.load("91.npz")

# def simulation_action(Kinova):
#     actions = np.array([
#         [0.15888406, 0.19621667, 0.10373455, -0.05561209, 0.03189043, 0.03438406, 0.999279],
#         [0.26643664, 0.34055474, 0.18410407, -0.01229728, 0.03462679, 0.03521782, 0.9999995],
#         [0.27574277, 0.34502602, 0.18616018, -0.0239279, 0.0478164, 0.03510898, 0.9999994],
#         [0.26804855, 0.33632004, 0.19110641, -0.02865765, 0.04276241, 0.03915315, 0.99999964],
#         [0.27531028, 0.33731747, 0.19269902, -0.03504151, 0.03917067, 0.04452255, 0.9999998],
#         [0.2783659, 0.3340348, 0.18992151, -0.03351122, 0.04249883, 0.04031506, 0.9999902],
#         [0.2781283, 0.33324888, 0.18835074, -0.03245223, 0.03658399, 0.04291222, 0.9999766],
#         [0.2809837, 0.32659385, 0.1880985, -0.03390574, 0.03661367, 0.04337214, 0.99994624],
#         [0.28222832, 0.32860756, 0.18485083, -0.03425875, 0.03701274, 0.04414301, 0.9999312],
#         [0.2817827, 0.3267043, 0.1862523, -0.03358232, 0.03229111, 0.04411081, 0.99992365],
#         [0.30116078, 0.3402489, 0.21377233, -0.0364955, 0.02157156, 0.0412526, 1.0],
#         [0.28921217, 0.34028313, 0.19684848, -0.03446257, 0.02495152, 0.04421294, 0.9998469],
#         [0.29049003, 0.34396437, 0.19856556, -0.03672194, 0.01769913, 0.04199235, 0.9998849],
#         [0.29012567, 0.34639177, 0.19824812, -0.03593228, 0.01450834, 0.04148229, 0.99991655],
#         [0.28787866, 0.34684077, 0.19927312, -0.03636366, 0.01260407, 0.04421016, 0.99987376],
#         [0.28865218, 0.3458338, 0.20126493, -0.03566457, 0.01285689, 0.04522399, 0.99986887],
#         [0.2908942, 0.34136596, 0.19543003, -0.0355466, 0.01219919, 0.04596032, 0.9999146],
#         [0.2884014, 0.3402575, 0.19883278, -0.03151531, 0.01081701, 0.04558214, 0.9999556],
#         [0.2833686, 0.33304742, 0.2031496, -0.02979312, 0.0223197, 0.04256041, 0.9999785],
#         [0.1114872, 0.12316544, 0.08845672, -0.01432372, 0.00945346, 0.01592402, 0.9994714],
#         [0.02172066, 0.01335007, -0.27762347, 0.00059047, -0.0022029972, -0.0087004453, 1.0],
#         [0.00831524, -0.01368463, -0.45245227, 0.00483032, -0.00439468, -0.00146686, 0.9995758],
#         [0.018789673, 0.00086333393, -0.45189235, 0.0013815061, -0.0040030694, -0.00024428032, 0.99936336],
#         [0.02304387, 0.01042221, -0.4478183, -0.00306197, 0.00139105, 0.0016872, 0.9995093],
#         [0.02437292, 0.01179846, -0.45010236, -0.00326889, 0.00168627, 0.00232295, 0.9993904],
#         [0.0254314, 0.01376798, -0.45280668, -0.00290287, 0.0017058, 0.00241324, 0.9993452],
#         [0.02534766, 0.01454904, -0.45446706, -0.00253601, 0.00113188, 0.00245543, 0.9991862],
#         [0.02470542, 0.01463942, -0.45109928, -0.00236048, 0.00273065, 0.00228226, 0.9990771],
#         [0.02392778, 0.01452815, -0.45051605, -0.00216011, 0.00330222, 0.00196455, 0.998921],
#         [0.02310509, 0.01474938, -0.45009148, -0.00244923, 0.00376249, 0.00158839, 0.99879944],
#         [0.025624717, 0.018105939, -0.44948867, -0.0025901927, 0.0000128783, 0.0015201739, 0.99999917],
#         [0.023353497, 0.01667043, -0.45622304, -0.0023089403, 0.0020593207, -0.00093333534, 0.99936932],
#         [0.024073567, 0.016190402, -0.45133629, -0.0022652054, 0.0033266607, 0.00099681795, 0.99979228],
#         [0.02258693, 0.01639017, -0.44829926, -0.00216337, 0.00347335, 0.00187689, 0.99974614],
#         [0.02313281, 0.01777384, -0.4567834, -0.00244953, 0.00218399, 0.00169209, 0.9995488],
#         [0.00967531, 0.010521358, -0.18915278, -0.0012685703, 0.00014169142, -0.001214235, 0.99908447],
#         [-0.0017232423, -0.00089057977, 0.0038223155, 0.00032689865, -0.00031888112, 0.00096787856, -0.99726707],
#         [-0.0025293219, -0.0018037678, 0.033343222, 0.00022737193, -0.00011058338, 0.0011616205, -0.99949712],
#         [-0.0027889779, -0.0010566848, 0.036694385, -0.00033123535, 0.00046053526, 0.00014331262, -0.99942344],
#         [-0.0021551724, -0.00027782936, 0.029285004, -0.00032511097, 0.00050972955, 0.00062600704, -0.99920583],
#         [-0.00055124069, -0.0018964219, 0.027963148, -0.00072195672, -0.00073434401, 0.0013794801, -0.99987996],
#         [-0.00039944632, -0.0014170222, 0.013525108, -0.000090087065, -0.00033886544, 0.0030670397, -0.99971426],
#         [-0.0023031791, 0.00039563418, 0.012090742, -0.0016994369, -0.00056647317, 0.0017386983, -0.99943596],
#         [-0.0019667528, -0.00022056792, 0.0066438308, -0.0019555758, 0.00094126322, 0.0023056616, -0.99901503],
#         [-0.0021943229, -0.00028546993, 0.0061588404, -0.0018328847, 0.00093328924, 0.0023913586, -0.99866939],
#         [-0.0017383782, 0.000094336458, 0.0052648373, -0.0022618803, 0.00098990055, 0.0025496166, -0.99869114],
#         [-0.0018621518, 0.00019208249, 0.0046312702, -0.0022422394, 0.00076763128, 0.0026293541, -0.99894333],
#         [-0.0014486745, -0.00054059172, 0.004096895, -0.0019660974, 0.0012560871, 0.0026343886, -0.99945557],
#         [-0.0017989156, -0.0013837256, 0.0032120834, -0.00062872184, 0.0027344979, 0.00051604415, -0.99950826],
#         [-0.04342791, 0.0198296, 0.04163105, -0.0014609, 0.00553921, 0.01061225, -0.99999994],
#     ])
#     for action in actions:
#         action = np.clip(action, -1., 1.)
#         Kinova.step(action)
#         time.sleep(0.5)



def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)



    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon



    # # Auto-fill camera rendering info if not specified
    # if args.camera_names is None:
    #     # We fill in the automatic values
    #     env_type = EnvUtils.get_env_type(env=env)
    #     args.camera_names = DEFAULT_CAMERAS[env_type]
    # if args.render:
    #     # on-screen rendering can only support one camera
    #     assert len(args.camera_names) == 1

    # need_pause = True
    # if need_pause:
    #     ans = input("continue? (y/n)")
    #     if ans != "y":
    #         exit()

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)



    # print("\n======== [Evaluation Arguments Summary] ========")
    # print(f"▶ Agent Checkpoint Path:          {args.agent}")
    # print(f"▶ Number of Rollouts:             {args.n_rollouts}")
    # print(f"▶ Target Category, Model, Link:   {args.eval_category_model_link_name}")
    # print(f"▶ Max Horizon (override):         {args.horizon}")
    # print(f"▶ Camera Names:                   {args.camera_names}")
    # print(f"▶ Output Dataset Path (.hdf5):    {args.dataset_path}")
    # print(f"▶ Save High-Dim Observations:     {args.dataset_obs}")
    # print(f"▶ Rollout Seed:                   {args.seed}")
    # print(f"▶ Output JSON Stats Path:         {args.json_path}")
    # print(f"▶ Error Log Path:                 {args.error_path}")
    # print("================================================\n")

    env_real_kinova = EnvRealKinova()
    # env_real_kinova.run_collection(2)
    # simulation_action(env_real_kinova)


    # exit()
    # policy.start_episode()

    # # reset
    # obs = env_real_kinova.get_observation()
    # act = policy(ob=obs)
    # act = np.clip(act, -1., 1.)
    # # next_obs = env_real_kinova.step(act)
    rollout_stats = []
    
    for i in tqdm(range(1)):
        try:
            for step_i in range(1000):
                policy.start_episode()
                obs = env_real_kinova.get_observation()
                act = policy(ob=obs)
                act = np.clip(act, -1., 1.)
                print(act)
                next_obs, reward, terminated, truncated, info = env_real_kinova.step(act)
                obs = next_obs
        except KeyboardInterrupt:
            if True:
                print("ctrl-C catched, stop execution")
                ans = input("success? (y / n)")
                rollout_stats.append((1 if ans == "y" else 0))
                print("*" * 50)
                print("have {} success out of {} attempts".format(np.sum(rollout_stats), len(rollout_stats)))
                print("*" * 50)
                continue
            else:
                sys.exit(0)
        
        if True:
            print("TERMINATE WITHOUT KEYBOARD INTERRUPT...")
            ans = input("success? (y / n)")
            rollout_stats.append((1 if ans == "y" else 0))
            continue
        rollout_stats.append(stats)

       
    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60. # total time taken for rollouts in minutes
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"]) # number of episodes attempted
    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if args.json_path is not None:
        json_f = open(args.json_path, "w")
        json_f.write(stats_json)
        json_f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # category, model, and link of the target asset that the policy will be evaluated on
    parser.add_argument(
        "--eval_category_model_link_name",
        type=str,
        default=None,
        help="(optional) comma-delimited category,model,link to evaluate on (for cabinet open task)",
    )

    # bounding box randomization along xyz axis during evaluation (in percentage)
    parser.add_argument(
        "--eval_bbox_rand",
        type=str,
        default=None,
        help="(optional) comma-delimited bounding box randomization during evaluation",
    )

    # position randomization along xyz axis during evaluation (in meter)
    parser.add_argument(
        "--eval_xyz_rand",
        type=str,
        default=None,
        help="(optional) comma-delimited xyz position randomization during evaluation",
    )

    # rotation randomization around local z-axis during evaluation (in radiance)
    parser.add_argument(
        "--eval_z_rot_rand",
        type=float,
        default=None,
        help="(optional) z-axis rotation randomization during evaluation in radiance",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
                it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
                observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
    )

    # Dump a file with the error traceback at this path. Only created if run fails with an error.
    parser.add_argument(
        "--error_path",
        type=str,
        default=None,
        help="(optional) dump a file with the error traceback at this path. Only created if run fails with an error.",
    )

    # TODO: clean up this arg
    # If provided, do not run actions in env, and instead just measure the rate of action computation
    parser.add_argument(
        "--hz",
        type=int,
        default=None,
        help="If provided, do not run actions in env, and instead just measure the rate of action computation and raise warnings if it dips below this threshold",
    )

    # TODO: clean up this arg
    # If provided, set num_inference_timesteps explicitly for diffusion policy evaluation
    parser.add_argument(
        "--dp_eval_steps",
        type=int,
        default=None,
        help="If provided, set num_inference_timesteps explicitly for diffusion policy evaluation",
    )

    args = parser.parse_args()

    # Process dataset
    run_trained_agent(args)
