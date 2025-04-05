from regex import F
import digital_cousins
# If you store the offline dataset elsewhere, please uncomment the following line and put the directory here
# digital_cousins.ASSET_DIR = "~/assets"

import os
from PIL import Image
import numpy as np
import torch
import argparse
from our_method.models.gpt import GPT
import our_method
import omnigibson as og

TEST_DIR = os.path.dirname(__file__)
SAVE_DIR = f"{TEST_DIR}/test_acdc_output"
TEST_IMG_PATH = f"{TEST_DIR}/test_img.png"
# TEST_IMG_PATH = f"{TEST_DIR}/test_img.jpg"
CAPTION = "Fridge. Cabinet. Car."


def test_acdc_step_1(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=True,
        run_step_2=False,
        run_step_3=False,
        step_1_output_path=None,
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline


def test_acdc_step_2(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=True,
        run_step_3=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipelin

def test_acdc_step_3(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    print(f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json")

    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=True,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline

def test_acdc_step_4(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4=True,
        run_step_5=False,
        run_step_6=False,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/acdc_output/step_3_output/step_3_output_info.json",
        # task_obj_output_path=f"{TEST_DIR}/acdc_output/task_output/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        goal_task = args.goal_task
    )
    del pipeline

def test_acdc_step_5(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4=False,
        run_step_5=True,
        run_step_6=False,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/acdc_output/step_3_output/step_3_output_info.json",
        task_object_extraction_output_path=f"{TEST_DIR}/acdc_output/task_object_extraction/target_object_extraction.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline

def test_acdc_step_6(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4=False,
        run_step_5=False,
        run_step_6=True,
        run_step_7=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/acdc_output/step_3_output/step_3_output_info.json",
        task_spatial_reasing_output_path=f"{TEST_DIR}/acdc_output/task_object_spatial_reasoning/task_obj_output_info.json",
        # task_obj_output_path=f"{TEST_DIR}/acdc_output/task_output/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline


def test_acdc_step_7(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()

    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        run_step_4=False,
        run_step_5=False,
        run_step_6=False,
        run_step_7=True,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/acdc_output/step_3_output/step_3_output_info.json",
        task_spatial_reasing_output_path=f"{TEST_DIR}/acdc_output/task_object_spatial_reasoning/task_obj_output_info.json",
        task_object_retrieval_path=f"{TEST_DIR}/acdc_output/task_object_retrieval/task_obj_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline

def test_task_proposals(args):
    from our_method.pipeline.acdc import ACDC
    pipeline = ACDC()

    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        step_3_output_path=f"{TEST_DIR}/acdc_output/step_3_output/step_3_output_info.json",
        task_proposals=True,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline

# OG test should always be at the end since it requires a full shutdown during termination
def test_og(args):
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    og.launch()

    print()
    print("*" * 30)
    print("All tests successfully completed!")
    print("*" * 30)
    print()
    og.shutdown()


def main(args):
    # Run all tests
    print()
    print("*" * 30)
    print("Starting tests...")
    print("*" * 30)
    print()

    # test_acdc_step_1(args)
    # test_acdc_step_2(args)
    # test_acdc_step_3(args)

    # test_task_proposals(args)
    # test_acdc_step_4(args) # Task Object Extraction
    # test_acdc_step_5(args) # Task Object Spatial Reasoning
    test_acdc_step_6(args) # Task Object Retrieval
    # test_acdc_step_7(args) # Task following Scene Generation
    # og.shutdown()

    # Final test -- OG should always come at the end
    # This og test cannot run together with test_acdc_step_3
    # because the simulator can only be launched once, and after calling og.shutdown(), the whole process will terminate
    # test_og(args)


if __name__ == "__main__":
    # Define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_api_key", type=str, required=True,
                        help="GPT API key to use. Must be compatible with GPT model specified")
    parser.add_argument("--gpt_version", type=str, default="4o", choices=list(GPT.VERSIONS.keys()),
                        help=f"GPT model version to use. Valid options: {list(GPT.VERSIONS.keys())}")
    parser.add_argument("--goal_task", type=str, default="Give me the cup in the cabinet above the microwave", help=f"User Guieded Task")

    args = parser.parse_args()

    main(args)
