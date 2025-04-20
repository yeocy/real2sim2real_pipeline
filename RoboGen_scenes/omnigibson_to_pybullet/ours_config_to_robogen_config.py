import numpy as np
import pybullet as p
import json
from scipy.spatial.transform import Rotation as R
import yaml
import os
from copy import deepcopy
import argparse

def get_transformation_matrix(position, orientation):
    """
    위치와 방향(쿼터니언 또는 오일러 각도)으로부터 4x4 변환 행렬을 계산합니다.
    
    :param position: [x, y, z] 형식의 위치 벡터
    :param orientation: [x, y, z, w] 형식의 쿼터니언 또는 [roll, pitch, yaw] 형식의 오일러 각도
    :return: 4x4 변환 행렬 (numpy array)
    """
    # 입력이 오일러 각도인지 쿼터니언인지 확인 (길이로 구분)
    if len(orientation) == 3:
        # 오일러 각도를 쿼터니언으로 변환
        quat = p.getQuaternionFromEuler(orientation)
    else:
        # 이미 쿼터니언 형태
        quat = orientation
    
    # 쿼터니언에서 3x3 회전 행렬 계산
    rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    
    # 4x4 변환 행렬 생성
    transform = np.identity(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position
    
    return transform

def ours_config_to_robogen_config(ours_base_dir, urdf_base_dir, robogen_config_path):
    ########################
    ## Our method configs ##
    ########################

    scene_info_json_filepath = f"{ours_base_dir}/task_scene_generation/scene_info.json"
    task_info_json_filepath = f"{ours_base_dir}/task_object_extraction_and_spatial_reasoning/task_obj_output_info_scenario_0.json"

    with open(scene_info_json_filepath, 'r') as f:
        scene_info_json_data = json.load(f)
    with open(task_info_json_filepath, 'r') as f:
        task_info_json_data = json.load(f)

    # print(scene_info_json_data.keys())
    # print(task_info_json_data.keys())

    # Camera
    cam_pose = scene_info_json_data['cam_pose']
    print(cam_pose)

    cam_transform = get_transformation_matrix(position=cam_pose[0], orientation=cam_pose[1])
    # cam_transform = get_transformation_matrix(position=[0, 0, 0.3], orientation=[0, -30, -30])
    print(cam_transform)

    position = cam_transform[:3, 3].tolist()
    # 회전 행렬 추출 (변환 행렬의 왼쪽 상단 3x3 부분)
    rotation_matrix = cam_transform[:3, :3]
    orientation = R.from_matrix(rotation_matrix).as_quat()
    print(f"camera position: {position}")
    print(f"camera orientation: {orientation}")

    # Objects
    objects = scene_info_json_data['objects']
    # target_objects = target_obj_json_data['objects']
    print(objects.keys())
    # print(target_objects.keys())

    obj_world_pose = dict()
    for obj_name, obj_val in objects.items():
        print(obj_val.keys())
        tf_from_cam = obj_val['tf_from_cam']

        transform_matrix = cam_transform @ tf_from_cam

        position = transform_matrix[:3, 3].tolist()

        # 회전 행렬 추출 (변환 행렬의 왼쪽 상단 3x3 부분)
        rotation_matrix = transform_matrix[:3, :3]
        orientation = R.from_matrix(rotation_matrix).as_quat()

        obj_world_pose[obj_name] = [position, orientation]

    object_configs = dict()
    for obj_name, obj_pose in obj_world_pose.items():
        obj_cfg = dict()
        obj_cfg['position'] = obj_pose[0]
        obj_cfg['orientation'] = obj_pose[1]
        obj_cfg['scale'] = objects[obj_name]['scale']
        obj_cfg['name'] = obj_name
        obj_cfg['movable'] = True

        obj_cfg['category'] = objects[obj_name]['category']
        obj_cfg['model'] = objects[obj_name]['model']

        object_configs[obj_name] = obj_cfg

        print(f"[{obj_name}]")
        print(f"position: {obj_pose[0]}")
        print(f"orientation: {obj_pose[1].tolist()}")
        print(f"scale: {objects[obj_name]['scale']}")



    ###########################
    ##  RoboGen config file  ##
    ###########################
    with open(robogen_config_path, 'r') as f:
        robogen_orig_config_ = yaml.safe_load(f)
    robogen_orig_config = {'obj_configs': []}
    for orig_config in deepcopy(robogen_orig_config_):
        if len(orig_config.keys()) > 2:
            print(f"orig_config.keys(): {orig_config.keys()}")
            robogen_orig_config['obj_configs'].append(orig_config)
        else:
            for cfg_name, cfg_val in orig_config.items():
                robogen_orig_config[cfg_name] = cfg_val
    print(f"robogen_orig_config: \n{robogen_orig_config}")

    robogen_task_config = []

    # table false
    robogen_task_config.append({'use_table': False})

    # Object info
    for obj_name, obj_cfg in object_configs.items():
        task_config = dict()
        task_config['center'] = f"({obj_cfg['position'][0]}, {obj_cfg['position'][1]}, {obj_cfg['position'][2]})"
        task_config['orientation'] = obj_cfg['orientation'].tolist()
        task_config['size'] = obj_cfg['scale']
        task_config['name'] = obj_cfg['name']
        task_config['type'] = 'custom_urdf'
        task_config['custom_urdf_path'] = os.path.join(urdf_base_dir, obj_cfg['category'], obj_cfg['model'], f"{obj_cfg['model']}.urdf")
        task_config['movable'] = obj_cfg['movable']

        robogen_task_config.append(task_config)

    # 기타 info
    task_name = robogen_orig_config['task_name']
    # task_name = task_info_json_data['task']
    task_description = robogen_orig_config['task_description']
    robogen_base_dir_ = "data/generated_task_from_ours"

    # -   solution_path: data/generated_task_from_description/MY_INSTRUCTION/task_MY_INSTRUCTION
    task_name_ = task_name.replace(" ", "_")
    # robogen_task_config.append({'solution_path': f"{robogen_base_dir_}/{task_name_}/task_{task_name_}"})
    robogen_task_config.append({'solution_path': robogen_orig_config['solution_path']})

    # # -   set_joint_angle_object_name: Bottle
    # robogen_task_config.append({'set_joint_angle_object_name': set_joint_angle_object_name})

    # # -   spatial_relationships:
    # #     - on, bottle, desk

    # robogen_task_config.append({'spatial_relationships': spatial_relationships})
    # -   task_description: MY TASK INTRUCTION DESCRIPTION
    #     task_name: MY INSTRUCTION
    robogen_task_config.append({'task_description': task_description,
                                'task_name': task_name})

    # # -   distractor_config_path: data/generated_task_from_description/MY_INSTRUCTION/MY_TASK_INSTRUCTION_DESCRIPTION.yaml
    # task_description_ = task_description.replace(' ', '_')
    # distractor_config_path = f"{robogen_base_dir_}/{task_name_}/{task_description_}.yaml"
    # robogen_task_config.append({'distractor_config_path': distractor_config_path})

    # Save
    # robogen_task_config_path = f"{robogen_base_dir}/{task_name_}/{task_name_}.yaml"
    robogen_task_config_path = os.path.join(os.path.dirname(robogen_config_path), f"_{os.path.basename(robogen_config_path)}")

    os.makedirs(os.path.dirname(robogen_task_config_path), exist_ok=True)

    with open(robogen_task_config_path, 'w') as f:
        yaml.dump(robogen_task_config, f, default_flow_style=False, sort_keys=False)
    print(f"Task config file saved at {robogen_task_config_path}")

    return cam_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_base_dir", type=str)
    parser.add_argument("--urdf_base_dir", type=str)
    parser.add_argument("--robogen_config_path", type=str)
    args = parser.parse_args()

    ours_config_to_robogen_config(ours_base_dir=args.ours_base_dir,
                                  urdf_base_dir=args.urdf_base_dir,
                                  robogen_config_path=args.robogen_config_path)