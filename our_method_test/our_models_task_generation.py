import os
import warnings

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
ONLY_STEP_7 = False

# 중간 산출물(step_1_output ~ task_scene_generation) 경로는 모두 args.save_dir 아래로
# 떨어진다. 예전에는 save_dir 과 무관하게 {TEST_DIR}/acdc_output 에서 읽도록 하드코딩돼
# 있어서, save_dir 을 바꾸면 step 1 은 새 위치에 쓰고 step 2 이후는 옛 위치에서 읽는
# 불일치가 생겼다. save_dir 은 configs/*.yaml 의 paths.save_dir 로 정한다.

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
    args.find_front_view = config['scene']['find_front_view']
    
    # Position randomization settings
    args.inside_position_randomization = config['position']['inside_position_randomization']
    # 그릇 안 배치 방식. center = 안쪽 bbox 한가운데, random = 안쪽에서 무작위,
    # keep = 앞 단계가 잡은 위치 그대로. 안 적으면 기존 동작(랜덤 켬/끔)을 따른다.
    args.inside_placement = config['position'].get(
        'inside_placement',
        'random' if config['position']['inside_position_randomization'] else 'keep')
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


def gaia_step_sam3d(args, config):
    """Step 1.5: 넣을 물체를 GPT 로 고르고(SceneObjectFilter), 그 물체를 SAM3D 로 만들어 풀을 짓는다.

    사람이 물체/프롬프트/인스턴스 대응을 적지 않는다. 결과 풀은 sam3d_pool_root(config, save_dir).
    """
    from our_method.models.gpt import GPT
    from our_method.pipeline.scene_object_filter import SceneObjectFilter
    from our_method.pipeline.sam3d_asset_generation import (
        SAM3DAssetGenerator, generator_call, sam3d_pool_root)
    from our_method.utils.asset_pool import use_pool_usd

    call = generator_call(config)
    step_1_output_path = f"{args.save_dir}/step_1_output/step_1_output_info.json"
    gpt = GPT(api_key=args.gpt_api_key, version=args.gpt_version, token_print=args.token_print,
              log_dir_tail="_SceneObjectFilter")
    discard = SceneObjectFilter(gpt)(step_1_output_path=step_1_output_path, goal_task=args.goal_task,
                                     save_dir=args.save_dir, use_cache=call.get("use_cache", True))
    det = json.load(open(json.load(open(step_1_output_path))["detected_categories"]))
    keep = [n for n in det["names"] if n not in discard]
    pool = SAM3DAssetGenerator()(
        step_1_output_path=step_1_output_path, keep_names=keep, save_dir=args.save_dir,
        inputs_dir=os.path.dirname(args.test_depth_img_path), pool_root=sam3d_pool_root(config, args.save_dir),
        image=os.path.join(TEST_DIR, call["image"]) if call.get("image") else None,
        gpus=call.get("gpus"), elevation=call.get("elevation", 40.0), title=args.goal_task,
        use_cache=call.get("use_cache", True))
    use_pool_usd(pool)


def gaia_step_2(args, config_path):
    """Run GAIA pipeline step 2: Digital cousin matching."""
    pipeline = GAIA(config=config_path)
    pipeline.run(
        input_path=args.test_img_path,
        save_dir=args.save_dir,
        run_step_1=False,
        run_step_2=True,
        run_step_3=False,
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
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
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{args.save_dir}/step_2_output/step_2_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        goal_task=args.goal_task,   # Step 2.5 SceneObjectFilter 가 태스크 물체를 keep 하는 데 쓴다
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
        run_step_7=False,
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{args.save_dir}/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{args.save_dir}/step_3_output/step_3_output_info.json",
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
        run_step_7=False,
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{args.save_dir}/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{args.save_dir}/step_3_output/step_3_output_info.json",
        task_spatial_reasoning_output_path=f"{args.save_dir}/task_object_extraction_and_spatial_reasoning/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        find_front_view=args.find_front_view,
        use_distractor_noise=args.use_distractor_noise,
        use_distractor_category=args.use_distractor_category,
        distractor_top_k=args.distractor_top_k
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
        run_step_7=True,
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{args.save_dir}/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{args.save_dir}/step_3_output/step_3_output_info.json",
        task_spatial_reasoning_output_path=f"{args.save_dir}/task_object_extraction_and_spatial_reasoning/task_obj_output_info.json",
        task_object_retrieval_path=f"{args.save_dir}/task_object_retrieval/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        gpt_token_print=args.token_print,
        find_front_view=args.find_front_view,
        inside_position_randomization=args.inside_position_randomization,
        inside_placement=args.inside_placement,
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
        step_1_output_path=f"{args.save_dir}/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{args.save_dir}/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{args.save_dir}/step_3_output/step_3_output_info.json",
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

    # 입력(사진, 깊이, camera_info 등)을 결과 폴더에 같이 남긴다: <save_dir>/inputs/
    import shutil
    input_dir = os.path.dirname(args.test_img_path)
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copytree(input_dir, os.path.join(args.save_dir, "inputs"), dirs_exist_ok=True)
    log.info(f"입력 복사: {input_dir} -> {os.path.join(args.save_dir, 'inputs')}")

    # Run steps based on config
    if steps.get('run_step_1', False):
        gaia_step_1(args, config_path)

    # Step 1.5: SAM3D 생성 모드면 Step 1 이 검출하고 GPT 가 남긴 물체를 SAM3D 로 만들어 풀을 짓는다
    from our_method.pipeline.sam3d_asset_generation import sam3d_enabled
    if sam3d_enabled(config) and steps.get('run_step_sam3d', True) and (
            steps.get('run_step_2', False) or steps.get('run_step_3', False)):
        gaia_step_sam3d(args, config)

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

    if steps.get('run_step_7', False):
        ran_heavy = any(steps.get(k, False) for k in
                        ('run_step_1', 'run_step_2', 'run_step_4_and_5', 'run_step_6'))
        if ran_heavy and not ONLY_STEP_7:
            # Step 6 의 FeatureMatcher(GSAM/DINO/CLIP) 가 남은 프로세스에서 Isaac 을 띄우면
            # RAM 이 모자라 OOM 으로 죽는다 (exit 137). Step 7 은 새 프로세스로 돌린다.
            import subprocess, sys
            log.info("Running Step 7 in a fresh process to free memory...")
            ret = subprocess.run([sys.executable, os.path.abspath(__file__),
                                  "--config", config_path, "--only_step_7"]).returncode
            # Isaac 은 종료 시 segfault(139) 를 내지만 결과는 정상이다.
            if ret not in (0, -11, 139):
                raise RuntimeError(f"Step 7 subprocess failed with exit code {ret}")
        else:
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
    
    parser.add_argument(
        "--only_step_7",
        action="store_true",
        help="config 의 pipeline_steps 를 무시하고 Step 7 만 돌린다 (Step 7 을 새 프로세스로 띄울 때 사용)"
    )
    
    cli_args = parser.parse_args()
    ONLY_STEP_7 = cli_args.only_step_7
    
    # Get absolute path for config
    config_path = os.path.join(TEST_DIR, cli_args.config) if not os.path.isabs(cli_args.config) else cli_args.config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))

    if ONLY_STEP_7:
        config['pipeline_steps'] = {'run_step_7': True}

    # usd_pool: true 면 asset_pool.root 풀의 og_dataset 에서 USD 를 먼저 불러온다 (경로 문자열을 줘도 된다)
    usd_pool = config.get('usd_pool')
    from our_method.pipeline.sam3d_asset_generation import sam3d_enabled
    if usd_pool and not sam3d_enabled(config):
        from our_method.utils.asset_pool import use_pool_usd
        use_pool_usd(config['asset_pool']['root'] if usd_pool is True else usd_pool)
    # SAM3D 생성 모드의 풀은 Step 1.5(gaia_step_sam3d)가 만들고 그때 use_pool_usd 를 부른다
    
    

    # Run main pipeline
    main(config, config_path)
