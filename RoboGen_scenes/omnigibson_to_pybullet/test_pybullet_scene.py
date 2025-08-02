import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d  # point cloud 처리 및 저장을 위한 라이브러리
import time
import sys
from scipy.spatial.transform import Rotation as R

from icecream import ic
ic.configureOutput(includeContext=True)

sys.path.append('/data/github_repos/RoboGen')
from manipulation.panda_static import PandaStatic

class RobotTeleoperationController:
    def __init__(self, robot):
        self.robot = robot
        self.target_position = np.array([0.5, 0, 0.5])  # 초기 목표 위치
        self.target_orientation = np.array([0, 0, 0])   # 초기 목표 방향 (오일러 각도)
        
        # 이동 속도 설정
        self.position_step = 0.01  # 위치 이동 스텝
        self.orientation_step = 0.05  # 방향 회전 스텝 (라디안)
        
        # 시각화 객체들
        self.target_sphere_id = None
        self.orientation_lines = []
        
        # 초기 EEF 위치를 목표 위치로 설정
        self.update_initial_target()
        self.create_target_visualization()
        
    def update_initial_target(self):
        """현재 EEF 위치를 초기 목표 위치로 설정"""
        eef_pos, eef_orn = self.robot.get_pos_orient(self.robot.right_end_effector)
        self.target_position = np.array(eef_pos)
        self.target_orientation = np.array(p.getEulerFromQuaternion(eef_orn))
        
    def create_target_visualization(self):
        """목표 위치와 방향을 시각화하는 구와 좌표축 생성"""
        # 목표 위치를 나타내는 빨간 구 생성
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1, 0, 0, 0.7]  # 반투명 빨간색
        )
        
        self.target_sphere_id = p.createMultiBody(
            baseMass=0,  # 정적 객체
            baseCollisionShapeIndex=-1,  # 충돌 없음
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.target_position
        )
        
    def update_target_visualization(self):
        """목표 위치와 방향 시각화 업데이트"""
        # 구 위치 업데이트
        target_quat = p.getQuaternionFromEuler(self.target_orientation)
        p.resetBasePositionAndOrientation(
            self.target_sphere_id, 
            self.target_position, 
            target_quat
        )
        
        # 기존 방향 표시 라인들 제거
        for line_id in self.orientation_lines:
            p.removeUserDebugItem(line_id)
        self.orientation_lines.clear()
        
        # 좌표축 그리기 (X: 빨강, Y: 초록, Z: 파랑)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(target_quat)).reshape(3, 3)
        
        # 각 축의 방향 벡터 계산
        axis_length = 0.1
        x_axis = rotation_matrix[:, 0] * axis_length
        y_axis = rotation_matrix[:, 1] * axis_length
        z_axis = rotation_matrix[:, 2] * axis_length
        
        # X축 (빨강)
        line_id = p.addUserDebugLine(
            self.target_position,
            [self.target_position[i] + x_axis[i] for i in range(3)],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )
        self.orientation_lines.append(line_id)
        
        # Y축 (초록)
        line_id = p.addUserDebugLine(
            self.target_position,
            [self.target_position[i] + y_axis[i] for i in range(3)],
            lineColorRGB=[0, 1, 0],
            lineWidth=3,
            lifeTime=0
        )
        self.orientation_lines.append(line_id)
        
        # Z축 (파랑)
        line_id = p.addUserDebugLine(
            self.target_position,
            [self.target_position[i] + z_axis[i] for i in range(3)],
            lineColorRGB=[0, 0, 1],
            lineWidth=3,
            lifeTime=0
        )
        self.orientation_lines.append(line_id)
        
    def handle_keyboard_input(self):
        """키보드 입력 처리 - 사용자 정의 키 사용"""
        keys = p.getKeyboardEvents()
        
        if not keys:
            return
            
        # 위치 제어 (u j h k n m)
        if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
            self.target_position[0] += self.position_step  # X축 전진 (u)
        if ord('j') in keys and keys[ord('j')] & p.KEY_WAS_TRIGGERED:
            self.target_position[0] -= self.position_step  # X축 후진 (j)
        if ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
            self.target_position[1] += self.position_step  # Y축 좌측 (h)
        if ord('k') in keys and keys[ord('k')] & p.KEY_WAS_TRIGGERED:
            self.target_position[1] -= self.position_step  # Y축 우측 (k)
        if ord('n') in keys and keys[ord('n')] & p.KEY_WAS_TRIGGERED:
            self.target_position[2] += self.position_step  # Z축 위로 (n)
        if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
            self.target_position[2] -= self.position_step  # Z축 아래로 (m)
            
        # 방향 제어 ([ ] ; ' . /)
        # Z축 제어 ([ ])
        if ord('[') in keys and keys[ord('[')] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([0, 0, self.orientation_step])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")
        if ord(']') in keys and keys[ord(']')] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([0, 0, -self.orientation_step])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")

        # Y축 제어 (; ')
        if ord(';') in keys and keys[ord(';')] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([0, self.orientation_step, 0])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")
        if ord("'") in keys and keys[ord("'")] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([0, -self.orientation_step, 0])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")
            
        # Z축 제어 (. /)
        if ord('.') in keys and keys[ord('.')] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([self.orientation_step, 0, 0])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")
        if ord('/') in keys and keys[ord('/')] & p.KEY_WAS_TRIGGERED:
            orientation_step = np.array([-self.orientation_step, 0, 0])
            orientation_step_R = R.from_euler('xyz', orientation_step, degrees=False)
            self.target_orientation = R.from_matrix(R.from_euler('xyz', self.target_orientation, degrees=False).as_matrix() @ orientation_step_R.as_matrix()).as_euler('xyz', degrees=False)
            print(f"self.target_orientation: {self.target_orientation}")

        self.target_orientation[self.target_orientation < np.pi]  += 2 * np.pi  # 오일러 각도를 0~2π 범위로 조정
        self.target_orientation[self.target_orientation > np.pi]  -= 2 * np.pi  # 오일러 각도를 -π~π 범위로 조정

            
        # 리셋 (스페이스바)
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            # self.reset_to_current_eef()
            self.print_status()
            
        # 그리퍼 제어 (o c)
        if ord('o') in keys and keys[ord('o')] & p.KEY_WAS_TRIGGERED:
            self.open_gripper()  # o: 그리퍼 열기
        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            self.close_gripper()  # c: 그리퍼 닫기
            
    def reset_to_current_eef(self):
        """현재 EEF 위치로 목표 위치 리셋"""
        self.update_initial_target()
        print(f"Target reset to current EEF position: {self.target_position}")
        
    def open_gripper(self):
        """그리퍼 열기"""
        gripper_positions = [0.04, 0.04]  # 열린 상태
        self.robot.set_gripper_open_position(
            self.robot.right_gripper_indices, 
            gripper_positions, 
            force=100
        )
        
    def close_gripper(self):
        """그리퍼 닫기"""
        self.robot.set_gripper_close_position(force=100)
        
    def control_robot(self):
        """로봇을 목표 위치로 제어"""
        try:
            # IK 계산
            target_quat = p.getQuaternionFromEuler(self.target_orientation)
            
            ik_solution = self.robot.ik(
                target_joint=self.robot.right_end_effector,
                target_pos=self.target_position,
                target_orient=target_quat,
                ik_indices=self.robot.right_arm_ik_indices,
                max_iterations=1000
            )
            
            # 로봇 제어
            if len(ik_solution) > 0:
                self.robot.control(
                    indices=self.robot.right_arm_joint_indices,
                    target_angles=ik_solution,
                    gains=self.robot.motor_gains,
                    # forces=self.robot.motor_forces
                    forces=5 * 240
                )
                
        except Exception as e:
            print(f"IK calculation failed: {e}")
            
    def print_status(self):
        """현재 상태 출력"""
        current_eef_pos, current_eef_orn = self.robot.get_pos_orient(self.robot.right_end_effector)
        current_euler = p.getEulerFromQuaternion(current_eef_orn)
        
        print(f"\n=== Robot EEF Status ===")
        print(f"Current EEF Position: {np.round(current_eef_pos, 3)}")
        print(f"Target Position:      {np.round(self.target_position, 3)}")
        print(f"Current EEF Orientation (Euler): {np.round(current_euler, 3)}")
        print(f"Target Orientation (Euler):      {np.round(self.target_orientation, 3)}")
        print(f"Position Error: {np.round(np.linalg.norm(np.array(current_eef_pos) - np.array(self.target_position)), 4)}")
        print(f"Target Pose: {(self.target_position.tolist(), self.target_orientation.tolist())}")
        
    def update(self):
        """메인 업데이트 루프"""
        self.handle_keyboard_input()
        self.update_target_visualization()
        self.control_robot()

    def viz_update(self):
        """시각화 업데이트"""
        self.handle_keyboard_input()
        self.update_target_visualization()
        # self.print_status()

        
    def move_robot_to_target(self, target_position, target_orientation):
        """로봇을 목표 위치와 방향으로 이동"""
        try:
            # IK 계산
            target_quat = p.getQuaternionFromEuler(target_orientation)
            
            ik_solution = robot.ik(
                target_joint=robot.right_end_effector,
                target_pos=target_position,
                target_orient=target_quat,
                ik_indices=robot.right_arm_ik_indices,
                max_iterations=1000
            )
            
            # 로봇 제어
            if len(ik_solution) > 0:
                robot.control(
                    indices=robot.right_arm_joint_indices,
                    target_angles=ik_solution,
                    gains=robot.motor_gains,
                    forces=robot.motor_forces
                )
                
        except Exception as e:
            print(f"IK calculation failed: {e}")

    def follow_trajectory(self, trajectory, steps=50, sleep=False):
        """로봇을 주어진 궤적에 따라 이동"""
        for trajectory_step in trajectory:
            self.target_position = np.array(trajectory_step[0])
            self.target_orientation = np.array(trajectory_step[1])

            self.move_robot_to_target(self.target_position, self.target_orientation)
            step_sim(steps=steps, sleep=sleep)


class DepthCamera:
    def __init__(self, width, height, fov, near, far, camera_pos=[0, 0, 0], target_pos=[0, 0, 0], up_vector=[0, 0, 1]):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = width / height

        self.fov_rad = np.deg2rad(self.fov)
        self.focal_length = (self.width / 2) / np.tan(self.fov_rad / 2)
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.view_matrix = None
        self.projection_matrix = None
        self.view_matrix_reshaped = None
        self.projection_matrix_reshaped = None

        self.camera_pos = None
        self.target_pos = None
        self.up_vector = None
        self.rgb_array = None
        self.depth_array = None
        self.seg_array = None

        self.set_camera(camera_pos=camera_pos, target_pos=target_pos, up_vector=up_vector)

    def set_camera(self, camera_pos, target_pos, up_vector):
        self.camera_pos = camera_pos
        self.target_pos = target_pos
        self.up_vector = up_vector

        # 카메라 매트릭스 계산
        self.view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.view_matrix_reshaped = np.array(self.view_matrix).reshape(4, 4)
        self.projection_matrix_reshaped = np.array(self.projection_matrix).reshape(4, 4)

    def render(self):
        # 카메라 이미지 가져오기
        print(f"view_matrix: {self.view_matrix}")
        print(f"projection_matrix: {self.projection_matrix}")
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            # flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        # numpy 배열로 변환
        self.rgb_array = np.array(rgb_img)
        self.depth_array = np.array(depth_img)
        self.seg_array = np.array(seg_img)

        return self.rgb_array, self.depth_array, self.seg_array

    def get_pointcloud(self):
        self.render()
        depthImg = self.depth_array

        # depth 이미지 처리 (원시 값은 0-1 사이의 클립된 값)
        depth_buffer = np.reshape(depthImg, [self.height, self.width])

        # 실제 거리(미터)로 변환
        far_plane = self.far
        near_plane = self.near
        depth_real = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth_buffer)

        # Point cloud 생성
        points = []
        colors = []
        # Point cloud 생성 - 벡터화된 방법으로 속도 향상
        # 좌표 격자 생성
        u_coords, v_coords = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()

        # depth 이미지를 1차원으로 평탄화
        depths = depth_real.flatten()

        # 유효한 깊이 값만 필터링
        valid_mask = (depths > self.near) & (depths < self.far)
        valid_u = u_coords[valid_mask]
        valid_v = v_coords[valid_mask]
        valid_depths = depths[valid_mask]
        ic(np.sum(valid_mask))

        # 픽셀 좌표를 3D 좌표로 변환 (벡터화된 연산)
        x = (valid_u - self.cx) * valid_depths / self.fx
        y = (valid_v - self.cy) * valid_depths / self.fy
        z = valid_depths

        # 카메라 좌표계의 점들 (x, y, z 축 순서)
        points_camera = np.vstack((x, y, z)).T

        # 카메라 회전을 고려한 변환
        camera_rotation = np.array([
            [self.view_matrix_reshaped[0, 0], self.view_matrix_reshaped[0, 1], self.view_matrix_reshaped[0, 2]],
            [self.view_matrix_reshaped[1, 0], self.view_matrix_reshaped[1, 1], self.view_matrix_reshaped[1, 2]],
            [self.view_matrix_reshaped[2, 0], self.view_matrix_reshaped[2, 1], self.view_matrix_reshaped[2, 2]]
        ])
        ic(camera_rotation)

        # 모든 점에 대해 한 번에 회전 적용 (행렬 곱)
        points_world_rel = np.dot(points_camera, camera_rotation.T)
        ic(points_world_rel[10])

        # 카메라 위치를 모든 점에 더하기 (벡터화된 연산)
        ic(self.camera_pos)
        points_world = points_world_rel + np.array(self.camera_pos)

        # RGB 색상 추출 (유효한 점에 대해서만)
        colors = self.rgb_array.reshape(-1, 4)[valid_mask] / 255.0

        # Open3D를 사용하여 point cloud 생성
        point_cloud = o3d.geometry.PointCloud()
        if len(points_world) > 0:
            point_cloud.points = o3d.utility.Vector3dVector(points_world)
            point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # RGB 색상만 사용

        return point_cloud

def print_joint_info(objectId):
    numJoints = p.getNumJoints(objectId)
    
    for i in range(numJoints):
        jointInfo = p.getJointInfo(objectId, i)
        
        print(f"--- Joint {i} ---")
        print(f"Joint Index: {jointInfo[0]}")
        print(f"Joint Name: {jointInfo[1].decode('utf-8')}")
        print(f"Joint Type: {jointInfo[2]}")  # 0:REVOLUTE, 1:PRISMATIC, 4:FIXED
        print(f"First Position Index: {jointInfo[3]}")
        print(f"First Velocity Index: {jointInfo[4]}")
        print(f"Flags: {jointInfo[5]}")
        print(f"Joint Damping: {jointInfo[6]}")
        print(f"Joint Friction: {jointInfo[7]}")
        print(f"Joint Lower Limit: {jointInfo[8]}")
        print(f"Joint Upper Limit: {jointInfo[9]}")
        print(f"Joint Max Force: {jointInfo[10]}")
        print(f"Joint Max Velocity: {jointInfo[11]}")
        print(f"Link Name: {jointInfo[12].decode('utf-8')}")
        print(f"Joint Axis: {jointInfo[13]}")
        print(f"Parent Frame Pos: {jointInfo[14]}")
        print(f"Parent Frame Orn: {jointInfo[15]}")
        print(f"Parent Index: {jointInfo[16]}")
        print()

def print_link_states(objectId):
    numJoints = p.getNumJoints(objectId)
    
    # Base link
    basePose = p.getBasePositionAndOrientation(objectId)
    print(f"Base Link - Position: {basePose[0]}, Orientation: {basePose[1]}")
    
    # 각 Joint에 연결된 Link들
    for i in range(numJoints):
        linkState = p.getLinkState(objectId, i)
        jointInfo = p.getJointInfo(objectId, i)
        linkName = jointInfo[12].decode('utf-8')
        
        print(f"Link {i} ({linkName}):")
        print(f"  World Position: {linkState[0]}")
        print(f"  World Orientation: {linkState[1]}")
        print(f"  Local Position: {linkState[2]}")
        print(f"  Local Orientation: {linkState[3]}")
        print()

def print_robot_structure(objectId):
    print("=== Robot Structure ===")
    numJoints = p.getNumJoints(objectId)
    
    # Base 정보
    print("Base Link (Root)")
    
    # Joint와 Link 트리 구조
    for i in range(numJoints):
        jointInfo = p.getJointInfo(objectId, i)
        jointName = jointInfo[1].decode('utf-8')
        linkName = jointInfo[12].decode('utf-8')
        parentIndex = jointInfo[16]
        jointType = jointInfo[2]
        
        # Joint 타입 이름
        type_names = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED", 2: "SPHERICAL", 3: "PLANAR"}
        typeName = type_names.get(jointType, "UNKNOWN")
        
        if parentIndex == -1:
            parent = "Base"
        else:
            parentJointInfo = p.getJointInfo(objectId, parentIndex)
            parent = parentJointInfo[12].decode('utf-8')
        
        print(f"├── Joint_{i}: {jointName} ({typeName})")
        print(f"│   └── Link: {linkName} (parent: {parent})")

def deactivate_collision(objectId):
    # 충돌 비활성화 (모든 링크와 충돌하지 않음)
    for i in range(p.getNumJoints(objectId) + 1):  # +1은 베이스 링크 포함
        if i == 0:
            link_id = -1  # 베이스 링크는 -1
        else:
            link_id = i - 1
        
        p.setCollisionFilterGroupMask(
            objectId, 
            link_id, 
            collisionFilterGroup=0,  # 그룹을 0으로 설정
            collisionFilterMask=0    # 마스크를 0으로 설정
        )

def load_robot(robot_name, robot_scale=1.0, use_suction=False, asset_dir=None, id=None, np_random=None):
    robot_classes = {
        # "panda": Panda,
        "panda_static": PandaStatic,
        # "sawyer": Sawyer,
        # "ur5": UR5,
    }
    robot_names = list(robot_classes.keys())
    robot_class = robot_classes[robot_name]
    
    # Create robot
    scaling = robot_scale  # x, y, z 방향으로 스케일
    robot = robot_class(robot_scale=scaling)
    robot.init(asset_dir, id, np_random, fixed_base=True, use_suction=use_suction)
    agents = [robot]
    suction_id = robot.right_gripper_indices[0]

    # Update robot motor gains
    robot.motor_gains = 0.05
    robot.motor_forces = 100.0

    # Set robot base position & orientation, and joint angles
    robot_base_pos = [0, -1, 0]
    robot_base_orient = [0, 0, 0, 1]
    robot_base_orient = robot_base_orient
    robot.set_base_pos_orient(robot_base_pos, robot_base_orient)
    init_joint_angles = [0 for _ in range(len(robot.right_arm_joint_indices))]
    robot.set_joint_angles(robot.right_arm_joint_indices, init_joint_angles)
    
    return robot

def print_controls():
    """조작법 출력"""
    print("\n" + "="*50)
    print("           ROBOT TELEOPERATION CONTROLS")
    print("="*50)
    print("Position Control:")
    print("  U/J : X축 전진/후진")
    print("  H/K : Y축 좌/우")
    print("  N/M : Z축 위/아래")
    print()
    print("Orientation Control:")
    print("  [/] : Roll (Z축 회전)")
    print("  ;/' : Pitch (Y축 회전)")
    print("  ./? : Yaw (X축 회전)")
    print()
    print("Gripper Control:")
    print("  O : 그리퍼 열기")
    print("  C : 그리퍼 닫기")
    print()
    print("Other:")
    print("  Space : 현재 EEF 위치로 타겟 리셋")
    print("="*50)
    print("빨간 구: 목표 위치")
    print("RGB 축: 목표 방향 (빨강=X, 초록=Y, 파랑=Z)")
    print("="*50)

def step_sim(steps=50, sleep=False):
    """시뮬레이션 스텝"""
    for i in range(steps):
        p.stepSimulation()
        if sleep:
            time.sleep(1/60)  # 60 FPS 기준으로 1/60초 대기

def move_robot_to_target(robot, target_position, target_orientation):
    """로봇을 목표 위치와 방향으로 이동"""
    try:
        # IK 계산
        target_quat = p.getQuaternionFromEuler(target_orientation)
        
        ik_solution = robot.ik(
            target_joint=robot.right_end_effector,
            target_pos=target_position,
            target_orient=target_quat,
            ik_indices=robot.right_arm_ik_indices,
            max_iterations=1000
        )
        
        # 로봇 제어
        if len(ik_solution) > 0:
            robot.control(
                indices=robot.right_arm_joint_indices,
                target_angles=ik_solution,
                gains=robot.motor_gains,
                forces=robot.motor_forces
            )
            
    except Exception as e:
        print(f"IK calculation failed: {e}")

def follow_trajectory(robot, trajectory, steps=50, sleep=False):
    """로봇을 주어진 궤적에 따라 이동"""
    for trajectory_step in trajectory:
        target_position = trajectory_step[0]
        target_orientation = trajectory_step[1]

        move_robot_to_target(robot, target_position, target_orientation)
        step_sim(steps=steps, sleep=sleep)

if __name__ == "__main__":
    # PyBullet 초기화
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for no GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # 바닥 생성
    planeId = p.loadURDF("plane.urdf")

    # 로봇 생성
    robot = load_robot(robot_name="panda_static", robot_scale=1.0, use_suction=False, asset_dir="/data/github_repos/RoboGen/manipulation/assets", id=physicsClient, np_random=np.random.RandomState(42))

    # 물체 로드 (예: 로봇이나 다른 물체)
    object_urdfs = {
        "refrigerator_orig": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/fridge/petcxr/petcxr.urdf",
        "refrigerator_2_1_1": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/fridge/petcxr/resize_petcxr_2_1_1.urdf",
        "refrigerator_2_1_1_copy": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/fridge/petcxr/resize_petcxr_2_1_1.urdf",
        "shelf_cabinet_0": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/top_cabinet/nitufd/nitufd.urdf",
        "shelf_cabinet_0_backup": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/top_cabinet/nitufd/nitufd_backup.urdf",
        "water_bottle": "/home/kodogyu/projects/Research/SATELLITE/real2sim2real_pipeline/our_method_test/urdf/bottle_of_water/cytqio/cytqio.urdf",
    }

    # objects를 로드하고 ID를 저장
    object_ids = {}
    for object_name, object_urdf in object_urdfs.items():
        obj_base = [0, 0, 0]  # 물체의 기본 위치
        obj_ori = [0, 0, 0, 1]  # 물체의 기본 방향 (쿼터니언)
        fixed_base = True  # 물체를 고정된 베이스로 로드

        if object_name == "refrigerator_orig":
            # obj_base = [0, 0, 1.2]  # 원래 크기의 냉장고 위치
            obj_base = [-2, 0, 1.2]  # 원래 크기의 냉장고 위치
            obj_ori = [0, 0, 0, 1]  # 원래 크기의 냉장고 방향
        if object_name == "refrigerator_2_1_1":
            obj_base = [-2, 2, 1.2]  # 원래 크기의 냉장고 위치
            obj_ori = [0, 0, 0, 1]  # 원래 크기의 냉장고 방향
        if object_name == "refrigerator_2_1_1_copy":
            obj_base = [-5, 2, 1.2]  # 원래 크기의 냉장고 위치
            obj_ori = [0, 0, 0, 1]  # 원래 크기의 냉장고 방향
            fixed_base = False
        elif object_name == "shelf_cabinet_0":
            obj_base = [0, 2, 0.6]  # 원래 크기의 냉장고 위치
            obj_ori = [-0.0027232124081984317, 0.0027268900913008276, -0.7015023703982114, 0.712656700319516]  # 원래 크기의 냉장고 방향
        elif object_name == "shelf_cabinet_0_backup":
            obj_base = [0, 0, 0.6]  # 원래 크기의 냉장고 위치
            obj_ori = [-0.0027232124081984317, 0.0027268900913008276, -0.7015023703982114, 0.712656700319516]  # 원래 크기의 냉장고 방향
        elif object_name == "water_bottle":
            obj_base = [-0.5, -0.5, 0.15]
            obj_ori = [0, 0, 0, 1]
            fixed_base = False

        object_ids[object_name] = p.loadURDF(object_urdf, obj_base, obj_ori, useFixedBase=fixed_base)

    # 물체 색상 변경
    rgb_colormap = [(0.7019607843137254, 0.8862745098039215, 0.803921568627451),
                    (0.9921568627450981, 0.803921568627451, 0.6745098039215687),
                    (0.796078431372549, 0.8352941176470589, 0.9098039215686274),
                    (0.9568627450980393, 0.792156862745098, 0.8941176470588236),
                    (0.9019607843137255, 0.9607843137254902, 0.788235294117647),
                    (1.0, 0.9490196078431372, 0.6823529411764706),
                    (0.9450980392156862, 0.8862745098039215, 0.8000000000000000),
                    (0.8, 0.8, 0.8)]

    for obj_idx, (name, objectId) in enumerate(object_ids.items()):
        rgb_color = rgb_colormap[obj_idx % len(rgb_colormap)]
        p.changeVisualShape(objectId, -1, rgbaColor=[rgb_color[0], rgb_color[1], rgb_color[2], 1])
        num_joints = p.getNumJoints(objectId)
        for i in range(num_joints):
            p.changeVisualShape(objectId, i, rgbaColor=[rgb_color[0], rgb_color[1], rgb_color[2], 1])

    # Trajectory 생성
    trajectory1 = [
        ([0.088, -0.68, 0.483], [3.142, -1.45, 1.535]),
        ([0.088, -0.68, 0.292], [3.142, -1.45, 1.535]),
        ([0.078, -0.36, 0.292], [3.142, -1.45, 1.535])
    ]
    trajectory2 = [
        ([0.468, -1.  ,   0.962], [3.142, -1.7 , 1.635]),
        ([0.468, -1.  ,   0.222], [3.142, -1.45, 1.535]),
        ([0.368, -0.44,   0.222], [3.142, -1.45, 1.535]),
        ([0.118, -0.26,   0.292], [3.142, -1.45, 1.535])
    ]
    trajectory3 = [
        ([0.088, -1.0000000000015328, 0.8424999999999999], [3.1415926535828684, 6.924814284398751e-12, 0.785398163397]),
        ([0.34800000000000014, -1.0000000000015328, 0.8424999999999999], [3.1415926535828684, 6.924814284398751e-12, 0.785398163397]),
        ([0.34800000000000014, -1.0000000000015328, 0.8424999999999999], [3.1415926535828684, 6.924814284398751e-12, 1.4853981633970006]),
        ([0.34800000000000014, -1.0000000000015328, 0.8424999999999999], [3.1415926535828684, -1.6499999999930761, 1.4853981633970006]),
        ([0.34800000000000014, -1.0000000000015328, 0.30249999999999944], [3.1415926535828684, -1.6499999999930761, 1.4853981633970006]),
        ([0.34800000000000014, -0.43000000000153227, 0.30249999999999944], [3.1415926535828684, -1.6499999999930761, 1.4853981633970006]),
        ([0.017999999999999967, -0.43000000000153227, 0.30249999999999944], [3.1415926535828684, -1.6499999999930761, 1.4853981633970006]),
        ([0.02799999999999997, -0.32000000000153217, 0.30249999999999944], [0.09999999999307363, -1.4915926535967197, -1.6561944901927905]),
        ([0.04799999999999997, -0.29000000000153214, 0.30249999999999944], [0.09999999999307363, -1.4915926535967197, -1.6561944901927905]),
        ([0.04799999999999997, -0.2500000000015321, 0.30249999999999944], [0.09999999999307363, -1.4915926535967197, -1.6561944901927905])
    ]

    # Teleoperation 컨트롤러 생성
    teleop_controller = RobotTeleoperationController(robot)
    
    # 조작법 출력
    print_controls()
    
    # 상태 출력 카운터
    status_counter = 0

    # Trajectory 따라 이동
    teleop_controller.open_gripper()  # 그리퍼 열기
    teleop_controller.follow_trajectory(trajectory3, steps=50, sleep=True)

    # 시뮬레이션 실행
    while True:
        ## Teleoperation 업데이트
        teleop_controller.update()

        # 주기적으로 상태 출력 (2초마다)
        status_counter += 1

        p.stepSimulation()
        time.sleep(1/60)  # 60 FPS

    # 종료
    p.disconnect()