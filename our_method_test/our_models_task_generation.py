import os
import sys
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress various warnings BEFORE importing other packages
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*resume_download.*')
warnings.filterwarnings('ignore', message='.*weights_only.*')
warnings.filterwarnings('ignore', message='.*torch.meshgrid.*')
warnings.filterwarnings('ignore', message='.*xFormers.*')

import argparse
import random
import yaml
import json

import numpy as np
import torch
import omnigibson as og
from loguru import logger as log
from rich.logging import RichHandler
from our_method.pipeline.gaia import GAIA

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

os.environ["OMNIGIBSON_HEADLESS"] = "1"

# Directory configuration
TEST_DIR = os.path.dirname(__file__)

def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(TEST_DIR, "configs", "default_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_args_from_config(config):
    """Convert config dictionary to args object."""
    class Args:
        pass
    
    args = Args()
    
    # Path settings
    args.save_dir = os.path.join(TEST_DIR, config['paths']['save_dir'])
    args.test_img_path = os.path.join(TEST_DIR, config['paths']['test_rgb_img_path'])
    # test_depth_img_path가 None이 아닌지 확인 후 처리
    if config['paths'].get('test_depth_img_path') is not None:
        args.test_depth_img_path = os.path.join(TEST_DIR, config['paths']['test_depth_img_path'])
    else:
        args.test_depth_img_path = None

    # Load camera intrinsics matrix if path is provided
    camera_intrinsics_path = config['paths'].get('camera_intrinsics_matrix_path')
    if camera_intrinsics_path is not None:
        camera_intrinsics_full_path = os.path.join(TEST_DIR, camera_intrinsics_path)
        if os.path.exists(camera_intrinsics_full_path):
            args.camera_intrinsics_matrix = load_camera_intrinsics(camera_intrinsics_full_path)
            log.info(f"Loaded camera intrinsics from {camera_intrinsics_full_path}")
        else:
            args.camera_intrinsics_matrix = None
    else:
        args.camera_intrinsics_matrix = None

    camera_info = config['paths'].get('camera_intrinsics_matrix_path')
    if camera_info is not None:
        camera_info_full_path = os.path.join(TEST_DIR, camera_info)
        if os.path.exists(camera_info_full_path):
            args.camera_info = load_camera_info(camera_info_full_path)
            log.info(f"Loaded camera info from {camera_info_full_path}")
        else:
            args.camera_info = None
    else:
        args.camera_info = None

    # GPT settings
    args.gpt_version = config['gpt']['version']
    args.token_print = config['gpt']['token_print']

    # Scene generation settings
    args.no_resizing = config['scene']['no_resizing']
    args.find_front_view = config['scene']['find_front_view']
    
    # Position randomization settings
    args.inside_position_randomization = config['position']['inside_position_randomization']
    args.max_bound = config['position']['max_bound']
    
    # Rotation randomization settings
    args.rotation_randomization = config['rotation']['rotation_randomization']
    args.random_degree = config['rotation']['random_degree']
    
    # Distractor settings
    args.use_distractor_noise = config['distractor']['use_distractor_noise']
    args.use_distractor_category = config['distractor']['use_distractor_category']
    args.distractor_top_k = config['distractor']['distractor_top_k']
    
    # Task specification
    args.goal_task = config['task']['goal_task']
    
    # Get OpenAI API key from environment
    args.gpt_api_key = os.getenv('OPENAI_API_KEY')
    if not args.gpt_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return args

def load_camera_intrinsics(json_path):
    """
    Load camera intrinsics matrix from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing camera information
        
    Returns:
        np.ndarray: 3x3 camera intrinsics matrix
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract intrinsics matrix
    intrinsics_matrix = np.array(data['intrinsics']['matrix'])
    
    return intrinsics_matrix

def load_camera_info(json_path):
    """
    Load camera intrinsics matrix from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing camera information
        
    Returns:
        np.ndarray: 3x3 camera intrinsics matrix
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def gaia_step_1(args, config_path):
    """Run GAIA pipeline step 1: Real-world extraction."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        input_depth_path=args.test_depth_img_path,
        camera_intrinsics_matrix=args.camera_intrinsics_matrix,
        save_dir=args.save_dir,
        run_step_1=True,
        run_step_2=False,
        run_step_3=False,
        step_1_output_path=None,
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print
    )
    del pipeline


def gaia_step_2(args, config_path):
    """Run GAIA pipeline step 2: Digital cousin matching."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=True,
        run_step_3=False,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print
    )
    del pipeline


def gaia_step_3(args, config_path):
    """Run GAIA pipeline step 3: Scene generation."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        camera_info=args.camera_info,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=True,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print
    )
    del pipeline


def gaia_step_4_and_5(args, config_path):
    """Run GAIA pipeline steps 4 & 5: Task object extraction and spatial reasoning."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4_and_5=True,
        run_step_6=False,
        run_task_object_resizing=False,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/gaia_output/step_3_output/step_3_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        goal_task=args.goal_task
    )
    del pipeline


def gaia_step_6(args, config_path):
    """Run GAIA pipeline step 6: Task object retrieval."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4_and_5=False,
        run_step_6=True,
        run_task_object_resizing=False,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/gaia_output/step_3_output/step_3_output_info.json",
        task_spatial_reasoning_output_path=f"{TEST_DIR}/gaia_output/task_object_extraction_and_spatial_reasoning/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        find_front_view=args.find_front_view,
        use_distractor_noise=args.use_distractor_noise,
        use_distractor_category=args.use_distractor_category,
        distractor_top_k=args.distractor_top_k
    )
    del pipeline


def gaia_object_resizing(args, config_path):
    """Run GAIA pipeline: Task object resizing."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4_and_5=False,
        run_step_6=False,
        run_task_object_resizing=True,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/gaia_output/step_3_output/step_3_output_info.json",
        task_spatial_reasoning_output_path=f"{TEST_DIR}/gaia_output/task_object_extraction_and_spatial_reasoning/task_obj_output_info.json",
        task_object_retrieval_path=f"{TEST_DIR}/gaia_output/task_object_retrieval/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        goal_task=args.goal_task,
        resizing=args.no_resizing,
        use_distractor_noise=args.use_distractor_noise,
    )
    del pipeline


def gaia_step_7(args, config_path):
    """Run GAIA pipeline step 7: Task-following scene generation."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4_and_5=False,
        run_step_6=False,
        run_task_object_resizing=False,
        run_step_7=True,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/gaia_output/step_3_output/step_3_output_info.json",
        task_spatial_reasoning_output_path=f"{TEST_DIR}/gaia_output/task_object_extraction_and_spatial_reasoning/task_obj_output_info.json",
        task_object_retrieval_path=f"{TEST_DIR}/gaia_output/task_object_retrieval/task_obj_output_info.json",
        task_object_resizing_path=f"{TEST_DIR}/gaia_output/task_object_resizing/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        find_front_view=args.find_front_view,
        resizing=args.no_resizing,
        inside_position_randomization=args.inside_position_randomization,
        max_bound=args.max_bound,
        rotation_randomization=args.rotation_randomization,
        random_degree=args.random_degree,
        use_distractor_noise=args.use_distractor_noise,
    )
    del pipeline


def run_visualize_scene(args, config_path):
    """Run scene visualization using GAIA pipeline's run_visualize_scene argument."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_visualize_scene=True,
        step_1_output_path=f"{TEST_DIR}/gaia_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/gaia_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/gaia_output/step_3_output/step_3_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
    )
    del pipeline


# OG test should always be at the end since it requires a full shutdown during termination
def test_og(args):
    """Test OmniGibson initialization. Should always be at the end since it requires full shutdown."""
    og.launch()
    log.success("All tests successfully completed!")
    og.shutdown()


def main(config, config_path):
    """Main function that runs the pipeline based on config."""
    # Convert config to args for compatibility with existing functions
    args = create_args_from_config(config)
    
    # Get pipeline steps from config
    steps = config['pipeline_steps']
    
    log.info("Starting GAIA Pipeline...")

    # Run steps based on config
    if steps.get('run_step_1', False):
        gaia_step_1(args, config_path)

    if steps.get('run_step_2', False):
        gaia_step_2(args, config_path)

    if steps.get('run_step_3', False):
        gaia_step_3(args, config_path)

    if steps.get('run_visualize_scene', False):
        run_visualize_scene(args, config_path)

    if steps.get('run_step_4_and_5', False):
        gaia_step_4_and_5(args, config_path)
    
    if steps.get('run_step_6', False):
        gaia_step_6(args, config_path)

    if steps.get('run_task_object_resizing', False):
        gaia_object_resizing(args, config_path)

    if steps.get('run_step_7', False):
        gaia_step_7(args, config_path)

    # Note: test_og() cannot run together with test_gaia_step_3()
    # because the simulator can only be launched once
    if steps.get('run_og_test', False):
        test_og(args)


if __name__ == "__main__":
    # Parse command-line arguments for config file path
    parser = argparse.ArgumentParser(description="GAIA Pipeline Task Generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file (default: configs/default_config.yaml)"
    )
    
    cli_args = parser.parse_args()
    
    # Get absolute path for config
    config_path = os.path.join(TEST_DIR, cli_args.config) if not os.path.isabs(cli_args.config) else cli_args.config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    

    # Run main pipeline
    main(config, config_path)
