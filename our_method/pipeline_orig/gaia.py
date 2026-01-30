"""
Top level entry point for GAIA
"""
from deps.OmniGibson.omnigibson.object_states import inside
import our_method
# If you store the offline dataset elsewhere, please uncomment the following line and put the directory here
# digital_cousins.ASSET_DIR = "~/assets"

import yaml
import argparse
import os
from copy import deepcopy
from loguru import logger as log
from our_method.models.feature_matcher import FeatureMatcher
from our_method.pipeline.extraction import RealWorldExtractor
from our_method.pipeline.matching import DigitalCousinMatcher
from our_method.pipeline.real_scene_generation import RealSceneGenerator
from our_method.pipeline.task_object_extraction_and_spatial_reasoning import TaskObjectExtractionAndSpatialReasoning
from our_method.pipeline.task_object_retrieval import TaskObjectRetrieval
from our_method.pipeline.task_scene_generation import TaskSceneGenerator
from our_method.pipeline.task_object_resizing import TaskObjectResizing
import omnigibson as og

class GAIA:
    """
    End-to-end pipeline for running GAIA
    """
    def __init__(self, config=None):
        """
        Args:
            config (None or str): Configuration to use when running GAIA. If None, will use default
                located at <PATH_TO_GAIA>/configs/default.yaml
        """
        # Load config if not specified
        config = f"{our_method.__path__[0]}/configs/default.yaml" if config is None else config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.config = config

    def run(
            self,
            input_path=None,
            input_depth_path=None,
            camera_intrinsics_matrix=None,
            save_dir=None,
            run_step_1=False,
            run_step_2=False,
            run_step_3=False,
            run_visualize_scene=False,
            run_step_4_and_5=False,
            run_step_6=False,
            run_step_7=False,
            run_task_object_resizing=False,
            step_1_output_path=None,
            step_2_output_path=None,
            step_3_output_path=None,
            task_spatial_reasoning_output_path=None,
            task_object_retrieval_path=None,
            task_object_resizing_path=None,
            gpt_api_key=None,
            gpt_version=None,
            gpt_token_print=None,
            goal_task = None,
            resizing = None,
            find_front_view = None,
            inside_position_randomization = None,
            max_bound = None,
            rotation_randomization = None,
            random_degree = None,
            use_distractor_noise = None,
            use_distractor_category=None,
            distractor_top_k=None,
    ):
        """
        Executes GAIA, running the following steps:
        1. Real World Extraction
        2. Digital Cousin Matching (per-object)
        3. Simulated Scene Generation

        Optionally skips some steps in case this run crashes mid-execution.

        Args:
            input_path (str): Absolute path to the input RGB image to use for GAIA
            save_dir (None or str): If specified save directory to use for GAIA. Otherwise, will create a directory
                called "acdc_output" in the same directory as @input_path. Note: save_dir should NOT be specified
                in the loaded config!
            run_step_1 (bool): Whether to run Step 1 or not
            run_step_2 (bool): Whether to run Step 2 or not
            run_step_3 (bool): Whether to run Step 3 or not
            step_1_output_path (None or str): If specified, the output path from Step 1 to use. This is only
                necessary if @run_step_1 is False and @run_step_2 is True
            step_2_output_path (None or str): If specified, the output path from Step 2 to use. This is only
                necessary if @run_step_2 is False and @run_step_3 is True
            gpt_api_key (None or str): If specified, the GPT API key to use (will override any value found in the
                loaded config)
            gpt_version (None or str): If specified, the GPT version to use (will override any value found in the
                loaded config)
        """
        # Copy config, and potentially overwrite GPT API key
        config = deepcopy(self.config)
        
        # Use provided save_dir or default to acdc_output in input_path directory
        if save_dir is None:
            save_dir = f"{os.path.dirname(input_path)}/acdc_output"
        
        # Cfg에 Save dir 설정
        for step in ["RealWorldExtractor", "DigitalCousinMatcher", "RealSceneGenerator", "TaskObjectExtractionAndSpatialReasoning", "TaskObjectRetrieval", "TaskObjectResizing", "TaskSceneGenerator"]:
            cur_save_dir = config["pipeline"][step]["call"].get("save_dir", None)
            assert cur_save_dir is None, f"save_dir should not be specified in {step} config! Got: {cur_save_dir}"
            config["pipeline"][step]["call"]["save_dir"] = save_dir

        # Cfg에 GPT 설정 - API key와 version 일괄 적용
        gpt_steps = ["RealWorldExtractor", "DigitalCousinMatcher", "TaskObjectExtractionAndSpatialReasoning", 
                     "TaskObjectRetrieval", "TaskObjectResizing", "TaskSceneGenerator"]
        
        if gpt_api_key is not None:
            for step in gpt_steps:
                config["pipeline"][step]["call"]["gpt_api_key"] = gpt_api_key
                
        if gpt_version is not None:
            for step in gpt_steps:
                config["pipeline"][step]["call"]["gpt_version"] = gpt_version

        if gpt_token_print is not None:
            for step in gpt_steps:
                config["pipeline"][step]["call"]["gpt_token_print"] = gpt_token_print

        # Goal task 설정
        config["pipeline"]["TaskObjectExtractionAndSpatialReasoning"]["call"]["goal_task"] = goal_task
        
        # FeatureMatcher 생성 (필요한 step들에서 사용)
        fm = None
        if run_step_1 or run_step_2 or run_step_6:
            fm = FeatureMatcher(**config["models"]["FeatureMatcher"])

        # Step 1: Real World Extraction
        if run_step_1:
            log.debug("Running GAIA: Step 1 -- Real World Extraction")
            
            step_1 = RealWorldExtractor(
                feature_matcher=fm,
                verbose=config["pipeline"]["verbose"],
            )
            
            # Prepare step 1 call arguments
            step_1_call_args = config["pipeline"]["RealWorldExtractor"]["call"].copy()
            step_1_call_args["input_path"] = input_path
            step_1_call_args["input_depth_path"] = input_depth_path
            
            # Override camera_intrinsics_matrix if provided
            if camera_intrinsics_matrix is not None:
                step_1_call_args["camera_intrinsics_matrix"] = camera_intrinsics_matrix
            
            success, step_1_output_path = step_1(**step_1_call_args)
            if not success:
                raise ValueError("Failed GAIA Step 1!")

        # Step 2: Digital Cousin Matching
        if run_step_2:
            log.debug("Running GAIA: Step 2 -- Digital Cousin Matching")
            
            step_2 = DigitalCousinMatcher(
                feature_matcher=fm,
                verbose=config["pipeline"]["verbose"],
            )
            success, step_2_output_path = step_2(
                step_1_output_path=step_1_output_path,
                **config["pipeline"]["DigitalCousinMatcher"]["call"],
            )
            if not success:
                raise ValueError("Failed GAIA Step 2!")

        # Step 3: Simulated Scene Generation
        if run_step_3:
            log.debug("Running GAIA: Step 3 -- Simulated Scene Generation")

            step_3 = RealSceneGenerator(
                verbose=config["pipeline"]["verbose"],
            )
            success, step_3_output_path = step_3(
                step_1_output_path=step_1_output_path,
                step_2_output_path=step_2_output_path,
                **config["pipeline"]["RealSceneGenerator"]["call"],
            )
            if not success:
                raise ValueError("Failed GAIA Step 3!")

        # Step Visualization (optional, using VisualizeScene class)
        if run_visualize_scene:
            from our_method.pipeline.visualize_scene import VisualizeScene
            vis_cfg = config["pipeline"]["VisualizeScene"]["call"]
            vis = VisualizeScene(
                step3_output_path=step_3_output_path,
                scene_idx=vis_cfg.get("scene_idx", 0),
                output=vis_cfg.get("output", None),
                n_render_steps=vis_cfg.get("n_render_steps", 1000),
                visual_only=vis_cfg.get("visual_only", False),
            )
            vis.run()

        # Steps 4&5: Task Object Extraction & Spatial Reasoning
        if run_step_4_and_5:
            log.debug("Running GAIA: Step 4&5 -- Task Object Extraction & Spatial Reasoning")

            step_4_and_5 = TaskObjectExtractionAndSpatialReasoning(
                verbose=config["pipeline"]["verbose"],
            )
            success, step_4_and_5_output_path = step_4_and_5(
                step_1_output_path=step_1_output_path,
                step_2_output_path=step_2_output_path,
                step_3_output_path=step_3_output_path,
                **config["pipeline"]["TaskObjectExtractionAndSpatialReasoning"]["call"],
            )
            if not success:
                raise ValueError("Failed GAIA Step 4&5!")

        # Step 6: Task Object Retrieval
        if run_step_6:
            log.debug("Running GAIA: Step 6 -- Task Object Retrieval")

            step_6 = TaskObjectRetrieval(
                feature_matcher=fm,
                verbose=config["pipeline"]["verbose"],
            )
            success, task_feature_matching_path = step_6(
                step_1_output_path=step_1_output_path,
                step_2_output_path=step_2_output_path,
                step_3_output_path=step_3_output_path,
                task_spatial_reasoning_output_path=task_spatial_reasoning_output_path,
                **config["pipeline"]["TaskObjectRetrieval"]["call"],
                find_front_view=find_front_view,
                use_distractor_noise=use_distractor_noise,
                use_distractor_category=use_distractor_category,
                distractor_top_k=distractor_top_k,              
            )
            if not success:
                raise ValueError("Failed GAIA Step 6!")
                
        # Task Object Resizing
        if run_task_object_resizing:
            log.debug("Running GAIA: Task Object Resizing")

            obj_resizing = TaskObjectResizing(
                verbose=config["pipeline"]["verbose"],
            )
            success, obj_resizing_output_path = obj_resizing(
                task_feature_matching_path=task_object_retrieval_path,
                **config["pipeline"]["TaskObjectResizing"]["call"],
                resizing=resizing,
                use_distractor_noise=use_distractor_noise,
            )
            if not success:
                raise ValueError("Failed Task Object Resizing!")

        # Step 7: Task-following Scene Generation
        if run_step_7:
            log.debug("Running GAIA: Step 7 -- Task-following Scene Generation")

            step_7 = TaskSceneGenerator(
                verbose=config["pipeline"]["verbose"],
            )
            success, step_7_output_path = step_7(
                step_1_output_path=step_1_output_path,
                step_2_output_path=step_2_output_path,
                step_3_output_path=step_3_output_path,
                task_feature_matching_path=task_object_resizing_path,
                **config["pipeline"]["TaskSceneGenerator"]["call"],
                find_front_view=find_front_view,
                resizing=resizing,
                inside_position_randomization=inside_position_randomization,
                max_bound=max_bound,
                rotation_randomization=rotation_randomization,
                random_degree=random_degree,
                use_distractor_noise=use_distractor_noise,
            )
            if not success:
                raise ValueError("Failed GAIA Step 7!")
                
def main(args):

    # Create GAIA and run
    pipeline = GAIA(config=args.config)
    pipeline.run(
        input_path=args.input_path,
        run_step_1=not args.skip_step_1,
        run_step_2=not args.skip_step_2,
        run_step_3=not args.skip_step_3,
        step_1_output_path=args.step_1_output_path,
        step_2_output_path=args.step_2_output_path,
        gpt_api_key=args.gpt_api_key,
    )
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Absolute path to input RGB file to use")
    parser.add_argument("--config", type=str, default=None,
                        help="Absolute path to config file to use. If not specified, will use default.")
    parser.add_argument("--gpt_api_key", type=str, default=None,
                        help="GPT API key to use. If not specified, will use value found from config file.")
    parser.add_argument("--skip_step_1", action="store_true",
                        help="If set, will skip GAIA Step 1 (Real World Extraction)")
    parser.add_argument("--skip_step_2", action="store_true",
                        help="If set, will skip GAIA Step 2 (Digital Cousin Matching)")
    parser.add_argument("--skip_step_3", action="store_true",
                        help="If set, will skip GAIA Step 3 (Simulated Scene Generation)")
    parser.add_argument("--step_1_output_path", type=str, default=None,
                        help="output path from Step 1 to use. Only necessary if --skip_step_1 is set and --skip_step_2 is not set.")
    parser.add_argument("--step_2_output_path", type=str, default=None,
                        help="output path from Step 2 to use. Only necessary if --skip_step_2 is set and --skip_step_3 is not set.")

    args = parser.parse_args()
    main(args)

