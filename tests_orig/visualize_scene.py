
import os
from PIL import Image
import numpy as np
import torch
import argparse
from digital_cousins.models.gpt import GPT
import digital_cousins
from digital_cousins.pipeline.visualize import SimulatedSceneVisualize
import omnigibson as og
import yaml

TEST_DIR = os.path.dirname(__file__)
SAVE_DIR = f"{TEST_DIR}/test_acdc_output"
TEST_IMG_PATH = f"{TEST_DIR}/test_img.png"
# TEST_IMG_PATH = f"{TEST_DIR}/test_img.jpg"
CAPTION = "Fridge. Cabinet. Car."

def test_acdc_step_3():
    config=None

    config = f"{digital_cousins.__path__[0]}/configs/default.yaml" if config is None else config
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    step_3 = SimulatedSceneVisualize(
        verbose=config["pipeline"]["verbose"],
    )
    success, step_3_output_path = step_3(
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        **config["pipeline"]["SimulatedSceneVisualize"]["call"],
    )
    if not success:
        raise ValueError("Failed ACDC Step 3!")

test_acdc_step_3()
og.shutdown()