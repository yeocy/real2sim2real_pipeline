import os
from ours_config_to_robogen_config import ours_config_to_robogen_config

#! Task
task = "Pick up the water bottle on the table"
object = 'water_bottle'

#! Directory
ours_base_dir = "<path/to/ours/acdc_output>"
robogen_base_dir = "<path/to/RoboGen>"
robogen_base_dir = f"{robogen_base_dir}/<path/to/robogen_task_config.yaml>"
urdf_base_dir = "<path/to/urdf_dir>"


def step1():
    # Run Subtask decomposition
    os.system(f"cd {robogen_base_dir} && \
              PYTHONPATH=$PYTHONPATH:{robogen_base_dir} \
              python gpt_4/prompts/prompt_from_description_ours.py --task '{task}' --object '{object}'")

def step2():
    # Transport Configurations
    cam_pose = ours_config_to_robogen_config(ours_base_dir=ours_base_dir, urdf_base_dir=urdf_base_dir, robogen_config_path=robogen_config_path)
    # os.system(f"python ours_config_to_robogen_config.py --ours_base_dir {ours_base_dir} --robogen_config_path {robogen_config_path}")
    print(f"cam_pose: {cam_pose}")
    cam_position_str = str(cam_pose[0]).replace(" ", "")
    cam_orientation_str = str(cam_pose[1]).replace(" ", "")
    print(f"cam_position_str: {cam_position_str}")
    print(f"cam_orientation_str: {cam_orientation_str}")

    # Run RoboGen skill learning pipeline
    os.system(f"cd {robogen_base_dir} && \
              python execute.py --task_config_path {robogen_config_path} \
              --gui 1 --run_training 1 \
              --table_id 0 --robot_name panda \
              --camera_position={cam_position_str} --camera_orientation {cam_orientation_str} \
              --use_gpt_spatial_relationship 0")


if __name__ == "__main__":
    # step1()
    step2()