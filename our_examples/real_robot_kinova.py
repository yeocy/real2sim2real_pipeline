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
from scipy.spatial.transform import Rotation as R
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

        # # RGB, Depth subscriber
        # self.rgb_sub = Subscriber(cam_rgb_topic, Image)
        # self.depth_sub = Subscriber(cam_depth_topic, Image)

        # # 정확한 시간 동기화: queue_size=10, slop=0.05s
        # self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=50, slop=0.15)
        # self.ts.registerCallback(self.synced_rgbd_callback)

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
    
    def get_current_ik_pose(self):
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
            # 쿼터니언 -> RPY (roll, pitch, yaw)
            curr_rot = R.from_quat(curr_quat).as_euler('xyz', degrees=False)
            print(curr_pos, curr_rot)
            return [curr_pos[0], curr_pos[1], curr_pos[2], curr_rot[0], curr_rot[1], curr_rot[2]]
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




def run_trained_agent(args):
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)




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

    print("Press Enter to move to the target...")
    count = 0
    # all_positions = [[0.017179907047241108, 0.2651591828301152, 0.1742899247150319], [0.01139342990803123, 0.2656367824369166, 0.16747464877599272], [0.006987102955833588, 0.26523867155041403, 0.16900632382142244], [0.006058404955382082, 0.266527670043186, 0.17684016451493834], [0.008322993746323379, 0.2668458694245801, 0.19104089786166967], [0.0122357463883883, 0.26684550608561786, 0.20559795443379703], [0.015263090329230747, 0.26696044673201325, 0.2200671561374049], [0.017072861527002034, 0.2667710173072593, 0.23452745614904724], [0.017365423602346455, 0.26604588606825486, 0.2489448741001593], [0.016091918218843995, 0.2647231319811525, 0.2633155271013107], [0.013396557449719834, 0.2627718473925864, 0.2776350837076479], [0.010899832851324495, 0.25977292031512556, 0.29189868293996357], [0.010250750662870498, 0.25595692771472156, 0.3061258916642512], [0.01171646304917269, 0.2515338181877578, 0.3202546282165818], [0.015167322094115768, 0.2464720395977973, 0.33421920301101105], [0.02058867958434092, 0.24077815310207695, 0.34799272439457096], [0.028018140140481776, 0.23450881302608506, 0.3615535091600436], [0.03735147324947981, 0.22765425780831117, 0.3748734038329443], [0.048500476490650576, 0.2202125278298972, 0.38791800104349006], [0.06102596574057241, 0.21225797276789615, 0.40062531595511686], [0.07409256644749988, 0.2038623025897428, 0.41300102509036984], [0.08736003213129058, 0.19494917482285823, 0.424991994825228], [0.10074531044607589, 0.18554702135426648, 0.43656259477013115], [0.11422685088441152, 0.17571490532185186, 0.44767661725504515], [0.12779747916907724, 0.1655275173112134, 0.4582972938477473], [0.14145182841681808, 0.1550600461038394, 0.46839418355766693], [0.15518556456985433, 0.144356973381488, 0.47797453329786277], [0.16899360188752172, 0.13345317556011604, 0.48705907370794377], [0.18286847815939006, 0.12241850356606054, 0.495639354147524], [0.19680214578797417, 0.1113402092318263, 0.5036982754664324], [0.21078661819323619, 0.10029992677673416, 0.5112298347091996], [0.22481344118065777, 0.08936477989274061, 0.5182418966510643], [0.23887422882820203, 0.0785910979678448, 0.5247512321416007], [0.2529611039883557, 0.06802834313731942, 0.5307787697944714], [0.26706681984624064, 0.05771859059169948, 0.536348707893432], [0.28118495137882216, 0.047696647114602087, 0.5414875519388781], [0.29530987980233897, 0.03799017240677838, 0.546223280227642], [0.30943668513561684, 0.028619874285891633, 0.5505846651007701], [0.32356152932304466, 0.019600256535732274, 0.554600463631089], [0.337680719425477, 0.010939828069524982, 0.5582992067378404], [0.3517909406456686, 0.0026413401621598886, 0.5617089283738687], [0.365890816111537, -0.005294313480591839, 0.5648552774192416], [0.37997913083928664, -0.012867743942458043, 0.5677615117629938], [0.39405465961088754, -0.020084309962088653, 0.5704501269902232], [0.40810969086489063, -0.026953074496582463, 0.572942923576656], [0.4221229109581169, -0.033486735807672696, 0.5752611712616978], [0.4360180205335065, -0.03970227900587675, 0.5774266920854928], [0.449573823597924, -0.04561876417693589, 0.5794606990960395], [0.4622459848495913, -0.05125653224882454, 0.5813829438775059], [0.4733363355427223, -0.05662457998968695, 0.5832119722322201], [0.48204741624093056, -0.06173245458333754, 0.5849657079010836], [0.4883199609916316, -0.0665928226484307, 0.5866605584101886], [0.49221090178456556, -0.07123060845482576, 0.5883088241966427], [0.49380795037450426, -0.07567196614961236, 0.5899180110461301], [0.493256152771048, -0.07993274842541154, 0.5914893471777717], [0.49065933644181886, -0.08402248013300606, 0.5930245819971057], [0.4860884415065065, -0.08795074072977105, 0.5945233780167155], [0.4796265909752157, -0.09172157184108465, 0.5959832459343365], [0.47213415750543136, -0.09533712267846117, 0.5974014955305883], [0.4659284210514942, -0.09880537136057721, 0.5987714122959887], [0.46184591213711096, -0.10213154804848967, 0.6000886589813937], [0.45982924529656394, -0.1053318201028266, 0.6013653914760072], [0.45974552518141426, -0.10842926214048565, 0.6026199425139543], [0.46147126092867735, -0.11143991099274908, 0.6038651707797672], [0.4633392767666848, -0.1143659058694586, 0.605105297416033], [0.4636760956678612, -0.11721634117366331, 0.6063395370655221], [0.46350117997172896, -0.11998250208715855, 0.6075587602008695], [0.46335548973957147, -0.1226579335108069, 0.6087549967453246], [0.46321856772714803, -0.12524183106440812, 0.60992695923693], [0.4629815943658164, -0.12773546312013795, 0.6110750826574937], [0.4628215408827339, -0.13014467799498686, 0.6122006914823287], [0.46276293191272216, -0.13247563290544817, 0.6133059395711914], [0.462584821688709, -0.13473294845561434, 0.614392716488925], [0.4620377341995103, -0.1369142942713617, 0.6154602625354576], [0.46074131406756696, -0.13901723265496502, 0.6165078777915957], [0.4587768163510883, -0.1410389756370236, 0.6175336767078795], [0.4569019284580586, -0.14298487952968464, 0.6185371305433688], [0.45575505990207316, -0.14486811448703962, 0.619521742336068], [0.455078238374041, -0.1467011337500046, 0.6204922380226364], [0.45446233885179593, -0.14849053177127458, 0.621450967800382], [0.45379309153380315, -0.15024081613562856, 0.6223987313009055], [0.4531457715524238, -0.15195460114024684, 0.6233355600391367], [0.45254777746354885, -0.15363441614704842, 0.624261625192302], [0.4518195119643656, -0.15528251529229986, 0.6251774494758696], [0.45076311076359504, -0.1569020315455325, 0.6260834606835021], [0.44953063256366704, -0.15849473498036337, 0.6269792297422391], [0.44833604475374617, -0.16006111431285164, 0.627864305027526], [0.44721156056189765, -0.1616028481475782, 0.6287387844322988], [0.44613925190902576, -0.16312202158223554, 0.629602869643609], [0.44509912196788054, -0.16462090512532868, 0.630456779296641], [0.44402264803168495, -0.1661031730667888, 0.631300997960215], [0.4426373688329563, -0.1675718312636858, 0.632136017829711], [0.44085465721814493, -0.16902823442357184, 0.6329615201534785], [0.4389464519672432, -0.17047099886157846, 0.6337768113118681], [0.43703221262896924, -0.17189947433298536, 0.6345816622386135], [0.43512085053455346, -0.17331417337556365, 0.6353760159776235], [0.4332422545215287, -0.17471605719393013, 0.6361598728811343], [0.4313899936895828, -0.17610735016392098, 0.6369334077059469], [0.4295473220390074, -0.17749029823348228, 0.6376967102311714], [0.42764947812998616, -0.17886599047824392, 0.6384496704508861], [0.42568589209284496, -0.18023599669876078, 0.6391921524701899], [0.42374424468844046, -0.1816002026044652, 0.6399240238883165], [0.421921936644118, -0.18295885120104782, 0.6406451725457417], [0.420350794848465, -0.18431101280763196, 0.6413555491815087], [0.4190474346930862, -0.18565703378887277, 0.6420551211643385], [0.4179847690747481, -0.1869991208061217, 0.642743777685619], [0.41719073493008757, -0.18833851688009284, 0.6434213734883797], [0.416633019201138, -0.18967590939682433, 0.6440877195636862], [0.41622073323965597, -0.19101116336110846, 0.6447426126667978], [0.4159344539577539, -0.19234400334712642, 0.6453859378417192], [0.41576851783005314, -0.19367229084729543, 0.6460178111737248], [0.4156531688249127, -0.19499488817448474, 0.6466382713940125], [0.41556678608825776, -0.19631217955780378, 0.6472472682521971], [0.41550540738193315, -0.19762420338980102, 0.6478447931284798], [0.4154242407975561, -0.19893111566735833, 0.6484307848990646], [0.4153227801313523, -0.20023338606598062, 0.6490052153731244], [0.415243054738963, -0.20153101344724633, 0.6495681657091131], [0.4151995024881982, -0.2028238872161019, 0.6501197468086167], [0.41515631478965526, -0.20411172740321315, 0.6506598756138159], [0.4150695797821448, -0.205394722364991, 0.6511885309714295]]
    bottle_position = [0.0, 0.45, 0.167]
    bottle_orientations = [-1.956339,  -0.125664,  0.157080]
    
    all_positions = [[0.017179907202914138, 0.46515918272004564, 0.174289925537354], [0.01139343006370426, 0.465636782326847, 0.16747464959831482], [0.006987103111506618, 0.4652386714403445, 0.16900632464374454], [0.006058405111055112, 0.46652766993311645, 0.17684016533726044], [0.008322993901996409, 0.46684586931451055, 0.19104089868399177], [0.01223574654406133, 0.4668455059755483, 0.20559795525611912], [0.015263090484903777, 0.4669604466219437, 0.220067156959727], [0.017072861682675064, 0.4667710171971897, 0.23452745697136934], [0.017365423758019485, 0.4660458859581853, 0.2489448749224814], [0.016091918374517025, 0.46472313187108294, 0.2633155279236328], [0.013396557605392864, 0.46277184728251686, 0.27763508452997], [0.010899833006997525, 0.459772920205056, 0.29189868376228567], [0.010250750818543528, 0.455956927604652, 0.3061258924865733], [0.01171646320484572, 0.4515338180776882, 0.3202546290389039], [0.015167322249788798, 0.44647203948772773, 0.33421920383333314], [0.02058867974001395, 0.4407781529920074, 0.34799272521689306], [0.028018140296154806, 0.4345088129160155, 0.3615535099823657], [0.03735147340515284, 0.4276542576982416, 0.3748734046552664], [0.048500476646323606, 0.42021252771982764, 0.38791800186581216], [0.06102596589624544, 0.4122579726578266, 0.40062531677743896], [0.07409256660317291, 0.40386230247967325, 0.41300102591269194], [0.08736003228696361, 0.3949491747127887, 0.4249919956475501], [0.10074531060174892, 0.38554702124419693, 0.43656259559245325], [0.11422685104008455, 0.3757149052117823, 0.44767661807736725], [0.12779747932475027, 0.36552751720114385, 0.4582972946700694], [0.1414518285724911, 0.35506004599376983, 0.46839418437998903], [0.15518556472552736, 0.34435697327141845, 0.47797453412018487], [0.16899360204319475, 0.3334531754500465, 0.48705907453026587], [0.1828684783150631, 0.322418503455991, 0.4956393549698461], [0.1968021459436472, 0.31134020912175675, 0.5036982762887545], [0.21078661834890922, 0.3002999266666646, 0.5112298355315217], [0.2248134413363308, 0.28936477978267106, 0.5182418974733864], [0.23887422898387506, 0.27859109785777525, 0.5247512329639228], [0.2529611041440287, 0.26802834302724987, 0.5307787706167935], [0.2670668200019137, 0.25771859048162993, 0.5363487087157541], [0.2811849515344952, 0.24769664700453253, 0.5414875527612002], [0.295309879958012, 0.23799017229670882, 0.5462232810499641], [0.30943668529128987, 0.22861987417582208, 0.5505846659230922], [0.3235615294787177, 0.21960025642566272, 0.554600464453411], [0.33768071958115004, 0.21093982795945543, 0.5582992075601625], [0.35179094080134166, 0.20264134005209034, 0.5617089291961908], [0.36589081626721004, 0.1947056864093386, 0.5648552782415637], [0.37997913099495967, 0.1871322559474724, 0.5677615125853159], [0.39405465976656057, 0.1799156899278418, 0.5704501278125453], [0.40810969102056366, 0.17304692539334798, 0.5729429243989781], [0.4221229111137899, 0.16651326408225775, 0.57526117208402], [0.43601802068917955, 0.1602977208840537, 0.5774266929078149], [0.44957382375359706, 0.15438123571299456, 0.5794606999183616], [0.4622459850052643, 0.1487434676411059, 0.581382944699828], [0.47333633569839534, 0.1433754199002435, 0.5832119730545422], [0.4820474163966036, 0.1382675453065929, 0.5849657087234057], [0.48831996114730464, 0.13340717724149975, 0.5866605592325107], [0.4922109019402386, 0.12876939143510469, 0.5883088250189648], [0.4938079505301773, 0.12432803374031809, 0.5899180118684522], [0.49325615292672104, 0.12006725146451891, 0.5914893480000938], [0.4906593365974919, 0.11597751975692439, 0.5930245828194278], [0.4860884416621795, 0.1120492591601594, 0.5945233788390376], [0.47962659113088874, 0.1082784280488458, 0.5959832467566586], [0.4721341576611044, 0.10466287721146927, 0.5974014963529104], [0.4659284212071672, 0.10119462852935324, 0.5987714131183108], [0.461845912292784, 0.09786845184144077, 0.6000886598037158], [0.45982924545223697, 0.09466817978710385, 0.6013653922983293], [0.4597455253370873, 0.0915707377494448, 0.6026199433362764], [0.4614712610843504, 0.08856008889718137, 0.6038651716020893], [0.4633392769223578, 0.08563409402047184, 0.6051052982383551], [0.46367609582353425, 0.08278365871626714, 0.6063395378878442], [0.463501180127402, 0.0800174978027719, 0.6075587610231916], [0.4633554898952445, 0.07734206637912355, 0.6087549975676467], [0.46321856788282106, 0.07475816882552233, 0.6099269600592521], [0.4629815945214894, 0.0722645367697925, 0.6110750834798158], [0.46282154103840695, 0.06985532189494359, 0.6122006923046508], [0.4627629320683952, 0.06752436698448228, 0.6133059403935135], [0.46258482184438204, 0.0652670514343161, 0.6143927173112471], [0.4620377343551833, 0.06308570561856874, 0.6154602633577797], [0.46074131422324, 0.06098276723496543, 0.6165078786139178], [0.45877681650676133, 0.058961024252906835, 0.6175336775302016], [0.45690192861373163, 0.05701512036024581, 0.6185371313656909], [0.4557550600577462, 0.05513188540289082, 0.6195217431583901], [0.455078238529714, 0.05329886613992585, 0.6204922388449585], [0.45446233900746896, 0.051509468118655866, 0.6214509686227041], [0.4537930916894762, 0.04975918375430188, 0.6223987321232276], [0.45314577170809683, 0.04804539874968361, 0.6233355608614588], [0.4525477776192219, 0.04636558374288202, 0.6242616260146241], [0.4518195121200386, 0.044717484597630586, 0.6251774502981917], [0.4507631109192681, 0.04309796834439794, 0.6260834615058242], [0.4495306327193401, 0.041505264909567074, 0.6269792305645612], [0.4483360449094192, 0.03993888557707881, 0.6278643058498481], [0.4472115607175707, 0.038397151742352253, 0.6287387852546209], [0.4461392520646988, 0.0368779783076949, 0.6296028704659311], [0.44509912212355357, 0.03537909476460177, 0.6304567801189631], [0.444022648187358, 0.03389682682314166, 0.6313009987825371], [0.4426373689886293, 0.03242816862624465, 0.6321360186520331], [0.44085465737381796, 0.030971765466358603, 0.6329615209758006], [0.4389464521229162, 0.029529001028351987, 0.6337768121341902], [0.43703221278464227, 0.02810052555694509, 0.6345816630609356], [0.4351208506902265, 0.026685826514366795, 0.6353760167999456], [0.43324225467720173, 0.025283942696000317, 0.6361598737034564], [0.43138999384525584, 0.023892649726009463, 0.636933408528269], [0.42954732219468045, 0.022509701656448167, 0.6376967110534935], [0.4276494782856592, 0.021134009411686527, 0.6384496712732082], [0.425685892248518, 0.019764003191169666, 0.639192153292512], [0.4237442448441135, 0.018399797285465258, 0.6399240247106386], [0.421921936799791, 0.017041148688882624, 0.6406451733680638], [0.420350795004138, 0.01568898708229849, 0.6413555500038308], [0.4190474348487592, 0.014342966101057675, 0.6420551219866606], [0.4179847692304211, 0.013000879083808758, 0.6427437785079411], [0.4171907350857606, 0.011661483009837603, 0.6434213743107018], [0.416633019356811, 0.01032409049310612, 0.6440877203860083], [0.416220733395329, 0.008988836528821986, 0.6447426134891199], [0.41593445411342694, 0.007655996542804022, 0.6453859386640413], [0.41576851798572617, 0.006327709042635021, 0.6460178119960469], [0.41565316898058574, 0.005005111715445709, 0.6466382722163346], [0.4155667862439308, 0.003687820332126668, 0.6472472690745192], [0.4155054075376062, 0.0023757965001294234, 0.6478447939508019], [0.4154242409532291, 0.0010688842225721196, 0.6484307857213867], [0.41532278028702535, -0.00023338617605017653, 0.6490052161954465], [0.41524305489463603, -0.0015310135573158856, 0.6495681665314352], [0.41519950264387123, -0.002823887326171448, 0.6501197476309388], [0.4151563149453283, -0.004111727513282704, 0.650659876436138], [0.41506957993781785, -0.005394722475060565, 0.6511885317937516]]
    all_positions = [bottle_position] + all_positions
    # all_orientations = [[-0.5330816524676848, 1.5431928152111896, 1.4258289377844562], [-0.3945541942068998, 1.5295115135559043, 1.5574654351887802], [-0.4902190012846922, 1.5207405931938442, 1.4616295813477889], [-0.6389638360283192, 1.5197986913565624, 1.3116922307324415], [-0.5000484574269838, 1.5191311561535872, 1.4487398316735818], [-0.3580574024265527, 1.517211331742116, 1.5884987543789557], [-0.23589649089111683, 1.514414019997481, 1.70700062563384], [-0.12737950508900994, 1.5108183642989652, 1.811606524926957], [-0.033220330253566896, 1.5065842334091686, 1.9018458622311876], [0.04751290187591048, 1.501955436545786, 1.9788630863476138], [0.11740542668478912, 1.497285252063019, 2.04548865417951], [0.18539932566859046, 1.4926713822436075, 2.112284423369414], [0.25066544943793007, 1.4881710832942523, 2.1777382099709843], [0.310939076207037, 1.4839179062963304, 2.2387649128580516], [0.3662759053560869, 1.479757376851508, 2.295307014978224], [0.4153311733888138, 1.4749826503931343, 2.345560795714468], [0.4559825060472342, 1.4691172043447835, 2.3869721599115983], [0.48868361568014934, 1.462412759767325, 2.4201523666728106], [0.515160905545957, 1.4551920837149597, 2.4469727714342064], [0.5354414685402982, 1.4478810105523992, 2.4672394205811945], [0.5518933767013434, 1.440457315746836, 2.482874127850388], [0.5659428368986266, 1.432864301402911, 2.4959202877170616], [0.5778996806888661, 1.425072116809039, 2.506792799031798], [0.5879970913359618, 1.4171019610782016, 2.5157380299737646], [0.5966369951055286, 1.4089962564330696, 2.523144336057446], [0.6043568040087318, 1.4008119929017626, 2.5295458569590332], [0.6112960382202984, 1.3925666371234366, 2.5350775421663587], [0.617399778119666, 1.384239762355309, 2.5396741577070188], [0.6227778449975827, 1.375849425759459, 2.5434296688643614], [0.6275922185148016, 1.3674332920713748, 2.5464879511812506], [0.6320402077106649, 1.3590068145311425, 2.5490332625262777], [0.6362418284388384, 1.350578060131678, 2.5511802899011116], [0.6402241749067826, 1.342155484752853, 2.5529544996355815], [0.6440228284645944, 1.33374332142512, 2.554393446225024], [0.6476690431430225, 1.3253442827791315, 2.555533416165706], [0.6511972239313282, 1.3169581946792048, 2.556416247007708], [0.6546374442114089, 1.308583758571236, 2.5570812270899315], [0.6579931788006829, 1.300223708785809, 2.557541769120527], [0.6613160798218369, 1.2918703305194916, 2.557859934842611], [0.6644836286463143, 1.2835525327861195, 2.5579210243091213], [0.6673434536513585, 1.2753095582342615, 2.5575781401586273], [0.6699289851152626, 1.267140661844095, 2.55687252728985], [0.6724521240917715, 1.258999139903151, 2.556032258457071], [0.6750181979984892, 1.2508589098581315, 2.555175563341211], [0.6775499871422573, 1.2427352146215638, 2.554244558946558], [0.6799725186166172, 1.2346386314910536, 2.553193627236456], [0.6821061912425623, 1.2265988175384965, 2.5518873447232724], [0.6836956397043639, 1.2186684073077556, 2.5500663586827557], [0.6842463558595311, 1.2109880064343383, 2.5470753945328517], [0.6834159571664193, 1.2036604751561502, 2.542838039751758], [0.6807482512499233, 1.1968337266231912, 2.5368727893975533], [0.6762917713515052, 1.1904804041528414, 2.529478526343282], [0.6700638113547214, 1.1845663660301193, 2.5207869298648298], [0.6623111758495566, 1.1789899338360352, 2.511140260692301], [0.6534176134203893, 1.173644099137415, 2.50093269890548], [0.6434691810685003, 1.1685166212267366, 2.490169063888307], [0.6326924329114325, 1.1635626783489408, 2.4789895481390913], [0.6215664751380582, 1.1586681258085259, 2.4677710423026404], [0.611045448435722, 1.1535764993715252, 2.4577860440449806], [0.6012473923900836, 1.1483340438269272, 2.448839642225916], [0.5915383193700171, 1.1432028659123783, 2.4397543672868722], [0.5814615510424288, 1.1382634300758685, 2.430107119352226], [0.5707459393540981, 1.133489873275745, 2.419962927758687], [0.5590851503706079, 1.1288995865958564, 2.4092424507456793], [0.5455060875020774, 1.1247535580450818, 2.3965176215196773], [0.5306504290935238, 1.120851080537955, 2.3826602124659053], [0.5155804137017407, 1.116939138595932, 2.3691729349439696], [0.5001709884866403, 1.1131096296505527, 2.3553740320612886], [0.48448762818600594, 1.1093530931678877, 2.3411709377892396], [0.4687097175276407, 1.1056197567652202, 2.3267422683127275], [0.4530927004977205, 1.101838876819675, 2.3124804026951145], [0.43742569018600796, 1.0980539775970404, 2.2980076170193904], [0.42187253414401565, 1.094216118542609, 2.2836100058045483], [0.40637076575293163, 1.0903506649609622, 2.269038540010006], [0.3911337498852928, 1.0864198012426223, 2.254534247641511], [0.3764877734001329, 1.0823753083936531, 2.2404495593559712], [0.36280616876560734, 1.0781577133236762, 2.2273228333519985], [0.3496353056411046, 1.0738523888940241, 2.2145271401767457], [0.33659963268313475, 1.0695089169765386, 2.2016862101469616], [0.3236612255576045, 1.0651205492549387, 2.1888743624242966], [0.31091717622570547, 1.0606655560725802, 2.1763044140240173], [0.2983831528433231, 1.0561440575918088, 2.163991177851812], [0.28603357258613515, 1.0515623139699106, 2.1518908566762125], [0.2737417271682188, 1.0469396733855625, 2.139816579754208], [0.26156931589534865, 1.0422630699098359, 2.1279163289868688], [0.2496540056204685, 1.0375157143510085, 2.116389924613056], [0.23796858869710383, 1.0327084706230583, 2.1051333406662933], [0.2264723636423908, 1.0278491762112565, 2.0940756280169337], [0.21515140865711915, 1.0229401588980664, 2.0832089070400053], [0.20398562736149609, 1.0179839545051825, 2.0725228241481224], [0.19290148781555172, 1.0129868895071854, 2.0619624065248385], [0.18174910077645817, 1.0079619926587693, 2.0513578787249744], [0.17072029886955614, 1.0028871076322137, 2.041039036720175], [0.15988855927146065, 0.9977609771248739, 2.03103249779464], [0.1492106966411793, 0.9925919873683116, 2.0212113927763444], [0.13869896981532404, 0.9873807070075147, 2.011585511677502], [0.12834782934817027, 0.9821291177781375, 2.002146235041729], [0.11807367385203626, 0.9768437182535084, 1.9928123129382707], [0.10786549553993646, 0.9715241820118985, 1.983602485736583], [0.09768816227613794, 0.9661731353568914, 1.9744735541525105], [0.08758958983740983, 0.9607874015399447, 1.965531704332672], [0.07755836451237828, 0.9553688766529671, 1.9567390490866503], [0.06763224654363852, 0.949916662008778, 1.9481403327593367], [0.057853204397813, 0.9444305720663118, 1.9397502069420025], [0.048162991097930645, 0.9389144350332326, 1.9314749923852095], [0.03854785714844891, 0.9333694118202138, 1.923349431861187], [0.028986083258358063, 0.9277971652270947, 1.91536613993238], [0.019435420648841028, 0.9221993860962154, 1.907476577680257], [0.009886956011641105, 0.9165764202873339, 1.8996716447065387], [0.0003840479859340607, 0.9109279286261951, 1.8920155437688868], [-0.009061455204713154, 0.9052531485026294, 1.8844657908293792], [-0.018432031186055263, 0.8995519724326337, 1.8770229597082284], [-0.027712701811511174, 0.8938254706658042, 1.8697195629743644], [-0.036913223812207754, 0.8880742992627959, 1.8625359837197273], [-0.046045013468144186, 0.8822990828793382, 1.8554575725328424], [-0.055087490271867184, 0.8765014634064578, 1.8485269616858604], [-0.06406264857622186, 0.8706808091620748, 1.8417071047572768], [-0.07301416112931682, 0.8648340415181592, 1.8349324391567692], [-0.08189350116456295, 0.8589649072876986, 1.828254918879857], [-0.09069292545959261, 0.8530748245682549, 1.8216936568280935]]
    all_orientations = [[-0.5330816524676848, 1.5431928152111896, 1.4258289377844562], [-0.3945541942068998, 1.5295115135559043, 1.5574654351887802], [-0.4902190012846922, 1.5207405931938442, 1.4616295813477889], [-0.6389638360283192, 1.5197986913565624, 1.3116922307324415], [-0.5000484574269838, 1.5191311561535872, 1.4487398316735818], [-0.3580574024265527, 1.517211331742116, 1.5884987543789557], [-0.23589649089111683, 1.514414019997481, 1.70700062563384], [-0.12737950508900994, 1.5108183642989652, 1.811606524926957], [-0.033220330253566896, 1.5065842334091686, 1.9018458622311876], [0.04751290187591048, 1.501955436545786, 1.9788630863476138], [0.11740542668478912, 1.497285252063019, 2.04548865417951], [0.18539932566859046, 1.4926713822436075, 2.112284423369414], [0.25066544943793007, 1.4881710832942523, 2.1777382099709843], [0.310939076207037, 1.4839179062963304, 2.2387649128580516], [0.3662759053560869, 1.479757376851508, 2.295307014978224], [0.4153311733888138, 1.4749826503931343, 2.345560795714468], [0.4559825060472342, 1.4691172043447835, 2.3869721599115983], [0.48868361568014934, 1.462412759767325, 2.4201523666728106], [0.515160905545957, 1.4551920837149597, 2.4469727714342064], [0.5354414685402982, 1.4478810105523992, 2.4672394205811945], [0.5518933767013434, 1.440457315746836, 2.482874127850388], [0.5659428368986266, 1.432864301402911, 2.4959202877170616], [0.5778996806888661, 1.425072116809039, 2.506792799031798], [0.5879970913359618, 1.4171019610782016, 2.5157380299737646], [0.5966369951055286, 1.4089962564330696, 2.523144336057446], [0.6043568040087318, 1.4008119929017626, 2.5295458569590332], [0.6112960382202984, 1.3925666371234366, 2.5350775421663587], [0.617399778119666, 1.384239762355309, 2.5396741577070188], [0.6227778449975827, 1.375849425759459, 2.5434296688643614], [0.6275922185148016, 1.3674332920713748, 2.5464879511812506], [0.6320402077106649, 1.3590068145311425, 2.5490332625262777], [0.6362418284388384, 1.350578060131678, 2.5511802899011116], [0.6402241749067826, 1.342155484752853, 2.5529544996355815], [0.6440228284645944, 1.33374332142512, 2.554393446225024], [0.6476690431430225, 1.3253442827791315, 2.555533416165706], [0.6511972239313282, 1.3169581946792048, 2.556416247007708], [0.6546374442114089, 1.308583758571236, 2.5570812270899315], [0.6579931788006829, 1.300223708785809, 2.557541769120527], [0.6613160798218369, 1.2918703305194916, 2.557859934842611], [0.6644836286463143, 1.2835525327861195, 2.5579210243091213], [0.6673434536513585, 1.2753095582342615, 2.5575781401586273], [0.6699289851152626, 1.267140661844095, 2.55687252728985], [0.6724521240917715, 1.258999139903151, 2.556032258457071], [0.6750181979984892, 1.2508589098581315, 2.555175563341211], [0.6775499871422573, 1.2427352146215638, 2.554244558946558], [0.6799725186166172, 1.2346386314910536, 2.553193627236456], [0.6821061912425623, 1.2265988175384965, 2.5518873447232724], [0.6836956397043639, 1.2186684073077556, 2.5500663586827557], [0.6842463558595311, 1.2109880064343383, 2.5470753945328517], [0.6834159571664193, 1.2036604751561502, 2.542838039751758], [0.6807482512499233, 1.1968337266231912, 2.5368727893975533], [0.6762917713515052, 1.1904804041528414, 2.529478526343282], [0.6700638113547214, 1.1845663660301193, 2.5207869298648298], [0.6623111758495566, 1.1789899338360352, 2.511140260692301], [0.6534176134203893, 1.173644099137415, 2.50093269890548], [0.6434691810685003, 1.1685166212267366, 2.490169063888307], [0.6326924329114325, 1.1635626783489408, 2.4789895481390913], [0.6215664751380582, 1.1586681258085259, 2.4677710423026404], [0.611045448435722, 1.1535764993715252, 2.4577860440449806], [0.6012473923900836, 1.1483340438269272, 2.448839642225916], [0.5915383193700171, 1.1432028659123783, 2.4397543672868722], [0.5814615510424288, 1.1382634300758685, 2.430107119352226], [0.5707459393540981, 1.133489873275745, 2.419962927758687], [0.5590851503706079, 1.1288995865958564, 2.4092424507456793], [0.5455060875020774, 1.1247535580450818, 2.3965176215196773], [0.5306504290935238, 1.120851080537955, 2.3826602124659053], [0.5155804137017407, 1.116939138595932, 2.3691729349439696], [0.5001709884866403, 1.1131096296505527, 2.3553740320612886], [0.48448762818600594, 1.1093530931678877, 2.3411709377892396], [0.4687097175276407, 1.1056197567652202, 2.3267422683127275], [0.4530927004977205, 1.101838876819675, 2.3124804026951145], [0.43742569018600796, 1.0980539775970404, 2.2980076170193904], [0.42187253414401565, 1.094216118542609, 2.2836100058045483], [0.40637076575293163, 1.0903506649609622, 2.269038540010006], [0.3911337498852928, 1.0864198012426223, 2.254534247641511], [0.3764877734001329, 1.0823753083936531, 2.2404495593559712], [0.36280616876560734, 1.0781577133236762, 2.2273228333519985], [0.3496353056411046, 1.0738523888940241, 2.2145271401767457], [0.33659963268313475, 1.0695089169765386, 2.2016862101469616], [0.3236612255576045, 1.0651205492549387, 2.1888743624242966], [0.31091717622570547, 1.0606655560725802, 2.1763044140240173], [0.2983831528433231, 1.0561440575918088, 2.163991177851812], [0.28603357258613515, 1.0515623139699106, 2.1518908566762125], [0.2737417271682188, 1.0469396733855625, 2.139816579754208], [0.26156931589534865, 1.0422630699098359, 2.1279163289868688], [0.2496540056204685, 1.0375157143510085, 2.116389924613056], [0.23796858869710383, 1.0327084706230583, 2.1051333406662933], [0.2264723636423908, 1.0278491762112565, 2.0940756280169337], [0.21515140865711915, 1.0229401588980664, 2.0832089070400053], [0.20398562736149609, 1.0179839545051825, 2.0725228241481224], [0.19290148781555172, 1.0129868895071854, 2.0619624065248385], [0.18174910077645817, 1.0079619926587693, 2.0513578787249744], [0.17072029886955614, 1.0028871076322137, 2.041039036720175], [0.15988855927146065, 0.9977609771248739, 2.03103249779464], [0.1492106966411793, 0.9925919873683116, 2.0212113927763444], [0.13869896981532404, 0.9873807070075147, 2.011585511677502], [0.12834782934817027, 0.9821291177781375, 2.002146235041729], [0.11807367385203626, 0.9768437182535084, 1.9928123129382707], [0.10786549553993646, 0.9715241820118985, 1.983602485736583], [0.09768816227613794, 0.9661731353568914, 1.9744735541525105], [0.08758958983740983, 0.9607874015399447, 1.965531704332672], [0.07755836451237828, 0.9553688766529671, 1.9567390490866503], [0.06763224654363852, 0.949916662008778, 1.9481403327593367], [0.057853204397813, 0.9444305720663118, 1.9397502069420025], [0.048162991097930645, 0.9389144350332326, 1.9314749923852095], [0.03854785714844891, 0.9333694118202138, 1.923349431861187], [0.028986083258358063, 0.9277971652270947, 1.91536613993238], [0.019435420648841028, 0.9221993860962154, 1.907476577680257], [0.009886956011641105, 0.9165764202873339, 1.8996716447065387], [0.0003840479859340607, 0.9109279286261951, 1.8920155437688868], [-0.009061455204713154, 0.9052531485026294, 1.8844657908293792], [-0.018432031186055263, 0.8995519724326337, 1.8770229597082284], [-0.027712701811511174, 0.8938254706658042, 1.8697195629743644], [-0.036913223812207754, 0.8880742992627959, 1.8625359837197273], [-0.046045013468144186, 0.8822990828793382, 1.8554575725328424], [-0.055087490271867184, 0.8765014634064578, 1.8485269616858604], [-0.06406264857622186, 0.8706808091620748, 1.8417071047572768], [-0.07301416112931682, 0.8648340415181592, 1.8349324391567692], [-0.08189350116456295, 0.8589649072876986, 1.828254918879857], [-0.09069292545959261, 0.8530748245682549, 1.8216936568280935]]
    all_orientations = [bottle_orientations] + all_orientations
    next_position = []
    while True:
        print(f"next_position: {next_position}")
        user_input = input(">> ")
        if user_input == "":
            # 빈 입력(Enter 키) 감지 → move_to_target 실행
            position = all_positions[count]
            orientation_rpy = all_orientations[count]  # roll, pitch, yaw in radians
            # orientation_rpy = [0, 3.14159, 1.5708]
            print("Moving to target pose:", position, orientation_rpy)
            
            # print("Waiting for 1 second")
            # rospy.sleep(1)
            next_position = all_positions[count + 1] if count + 1 < len(all_positions) else None

            env_real_kinova.move_to_pose_target(position, orientation_rpy)
            print(env_real_kinova.get_current_ik_pose())
            count += 1
        else:
            print("Press Enter to move. You typed something else.")
            break

    

    # def move_to_pose_target(self, position, orientation_rpy):
    #     """ 
    #     Move to a desired pose using IK.

    #     Args:
    #         position (list): [x, y, z]
    #         orientation_rpy (list): [roll, pitch, yaw] in radians
    #     """
    #     if not self.is_init_success:
    #         rospy.logerr("MoveIt not initialized.")
    #         return False

    #     pose_target = Pose()
    #     pose_target.position.x = position[0]
    #     pose_target.position.y = position[1]
    #     pose_target.position.z = position[2]

    #     q = quaternion_from_euler(*orientation_rpy)
    #     pose_target.orientation.x = q[0]
    #     pose_target.orientation.y = q[1]
    #     pose_target.orientation.z = q[2]
    #     pose_target.orientation.w = q[3]

    #     pose_stamped = PoseStamped()
    #     pose_stamped.header.frame_id = self.robot.get_planning_frame()
    #     pose_stamped.header.stamp = rospy.Time.now()
    #     pose_stamped.pose = pose_target

    #     self.arm_group.set_pose_target(pose_stamped)
    #     # set_pose_target : 현재 위치에서 Target Pose
    #     success = self.arm_group.go(wait=True)

    #     self.arm_group.stop()
    #     self.arm_group.clear_pose_targets()

    #     if not success:
    #         rospy.logerr("Failed to move to desired pose.")
    #     else:
    #         rospy.loginfo("Successfully moved to desired pose.")
    #     return success

    exit()
    # simulation_action(env_real_kinova)

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
