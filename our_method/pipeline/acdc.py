"""
Top level entry point for ACDC
"""
import our_method
# If you store the offline dataset elsewhere, please uncomment the following line and put the directory here
# digital_cousins.ASSET_DIR = "~/assets"

import yaml
import argparse
import os
from copy import deepcopy
from our_method.models.feature_matcher import FeatureMatcher
from our_method.pipeline.extraction import RealWorldExtractor
from our_method.pipeline.matching import DigitalCousinMatcher
from our_method.pipeline.real_scene_generation import RealSceneGenerator

from our_method.pipeline.task_proposals import TaskProposals
from our_method.pipeline.task_object_extraction import TaskObjectExtraction
from our_method.pipeline.task_object_spatial_reasoning import TaskObjectSpatialReasoning
from our_method.pipeline.task_object_retrieval import TaskObjectRetrieval
from our_method.pipeline.task_scene_generation import TaskSceneGenerator
from our_method.pipeline.task_object_resizing import TaskObjectResizing
import omnigibson as og

class ACDC:
    """
    End-to-end pipeline for running ACDC
    """
    def __init__(self, config=None):
        """
        Args:
            config (None or str): Configuration to use when running ACDC. If None, will use default
                located at <PATH_TO_ACDC>/configs/default.yaml
        """
        # Load config if not specified
        config = f"{our_method.__path__[0]}/configs/default.yaml" if config is None else config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config

    def run(
            self,
            input_path=None,
            save_dir=None,
            run_step_1=False,
            run_step_2=False,
            run_step_3=False,
            run_step_4=False,
            run_step_5=False,
            run_step_6=False,
            run_step_7=False,
            run_task_object_resizing=False,
            task_proposals=False,
            step_1_output_path=None,
            step_2_output_path=None,
            step_3_output_path=None,
            task_object_extraction_output_path=None,
            task_spatial_reasing_output_path=None,
            task_object_retrieval_path=None,
            task_object_resizing_path=None,
            gpt_api_key=None,
            gpt_version=None,
            goal_task = None
    ):
        """
        Executes ACDC, running the following steps:
        1. Real World Extraction
        2. Digital Cousin Matching (per-object)
        3. Simulated Scene Generation

        Optionally skips some steps in case this run crashes mid-execution.

        Args:
            input_path (str): Absolute path to the input RGB image to use for ACDC
            save_dir (None or str): If specified save directory to use for ACDC. Otherwise, will create a directory
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
        save_dir = f"{os.path.dirname(input_path)}/acdc_output"
        # Cfg에 Save dir 설정
        for step in ["RealWorldExtractor", "DigitalCousinMatcher", "RealSceneGenerator", "TaskProposals",
                     "TaskObjectExtraction", "TaskObjectSpatialReasoning", "TaskObjectRetrieval", "TaskObjectResizing", "TaskSceneGenerator"]:
            cur_save_dir = config["pipeline"][step]["call"].get("save_dir", None)
            assert cur_save_dir is None, f"save_dir should not be specified in {step} config! Got: {cur_save_dir}"
            config["pipeline"][step]["call"]["save_dir"] = save_dir
        # Cfg에 GPT 설정
        if gpt_api_key is not None:
            config["pipeline"]["RealWorldExtractor"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["DigitalCousinMatcher"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["TaskObjectExtraction"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["TaskObjectSpatialReasoning"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["TaskObjectRetrieval"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["TaskObjectResizing"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["TaskProposals"]["call"]["gpt_api_key"] = gpt_api_key
        if gpt_version is not None:
            config["pipeline"]["RealWorldExtractor"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["DigitalCousinMatcher"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["TaskObjectExtraction"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["TaskObjectSpatialReasoning"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["TaskObjectRetrieval"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["TaskObjectResizing"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["TaskProposals"]["call"]["gpt_version"] = gpt_version
        config["pipeline"]["TaskObjectExtraction"]["call"]["goal_task"] = goal_task
        print(f"""

{"#" * 50}
{"#" * 50}
# Starting ACDC!
{"#" * 50}
{"#" * 50}

        """)
        if run_step_1 or run_step_2 or run_step_6:
            # We are running at least step 1 or step 2, so create FeatureMatcher
            fm = FeatureMatcher(**config["models"]["FeatureMatcher"])

            # Create RealWorldExtractor and run
            if run_step_1:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 1 -- Real World Extraction
{"#" * 50}
{"#" * 50}

                        """)
                step_1 = RealWorldExtractor(
                    feature_matcher=fm,
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_1_output_path = step_1(
                    input_path=input_path,
                    **config["pipeline"]["RealWorldExtractor"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 1!")

            if run_step_2:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 2 -- Digital Cousin Matching
{"#" * 50}
{"#" * 50}

                        """)
                step_2 = DigitalCousinMatcher(
                    feature_matcher=fm,
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_2_output_path = step_2(
                    step_1_output_path=step_1_output_path,
                    **config["pipeline"]["DigitalCousinMatcher"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 2!")

        if run_step_3:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 3 -- Simulated Scene Generation
{"#" * 50}
{"#" * 50}

                        """)

                step_3 = RealSceneGenerator(
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_3_output_path = step_3(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    **config["pipeline"]["RealSceneGenerator"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 3!")
        
        if task_proposals:  

                print(f"""

{"#" * 50}
{"#" * 50}
# Running Task Proposals
{"#" * 50}
{"#" * 50}

                        """)

                task_proposal = TaskProposals(
                    verbose=config["pipeline"]["verbose"],
                )
                success, task_proposals_path = task_proposal(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    step_3_output_path=step_3_output_path,
                    **config["pipeline"]["TaskProposals"]["call"],
                )
                if not success:
                    raise ValueError("Failed TaskProposal")

        if run_step_4:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running Task Object Extraction
{"#" * 50}
{"#" * 50}

                        """)

                step_4 = TaskObjectExtraction(
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_4_output_path = step_4(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    step_3_output_path=step_3_output_path,
                    **config["pipeline"]["TaskObjectExtraction"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 4!")
                
        if run_step_5:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running Task Object Spatial Reasoning
{"#" * 50}
{"#" * 50}

                        """)
                step_5 = TaskObjectSpatialReasoning(
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_5_output_path = step_5(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    step_3_output_path=step_3_output_path,
                    task_object_extraction_output_path= task_object_extraction_output_path,
                    **config["pipeline"]["TaskObjectSpatialReasoning"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 5!")


        if run_step_6:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 6 -- Simulated Scene Generation
{"#" * 50}
{"#" * 50}

                        """)

                step_6 = TaskObjectRetrieval(
                    feature_matcher=fm,
                    verbose=config["pipeline"]["verbose"],
                )
                success, task_feature_matching_path = step_6(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    step_3_output_path=step_3_output_path,
                    task_spatial_reasing_output_path = task_spatial_reasing_output_path,
                    **config["pipeline"]["TaskObjectRetrieval"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 6!")
                
        if run_task_object_resizing:
             
                print(f"""

{"#" * 50}
{"#" * 50}
# Running SATELLITE: Resizing the Task Objects
{"#" * 50}
{"#" * 50}

                        """)

                obj_resizing = TaskObjectResizing(
                    verbose=config["pipeline"]["verbose"],
                )
                success, obj_resizing_output_path = obj_resizing(
                    task_feature_matching_path = task_object_retrieval_path,
                    **config["pipeline"]["TaskObjectResizing"]["call"],
                )
                if not success:
                    raise ValueError("Failed SATELLITE Task Object Resizing!")

        if run_step_7:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 7 -- Simulated Scene Generation
{"#" * 50}
{"#" * 50}

                        """)

                step_7 = TaskSceneGenerator(
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_7_output_path = step_7(
                    step_1_output_path=step_1_output_path,
                    step_2_output_path=step_2_output_path,
                    step_3_output_path=step_3_output_path,
                    task_feature_matching_path = task_object_resizing_path,
                    # task_feature_matching_path = task_object_retrieval_path,
                    **config["pipeline"]["RealSceneGenerator"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 7!")
                
def main(args):

    # Create ACDC and run
    pipeline = ACDC(config=args.config)
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
                        help="If set, will skip ACDC Step 1 (Real World Extraction)")
    parser.add_argument("--skip_step_2", action="store_true",
                        help="If set, will skip ACDC Step 2 (Digital Cousin Matching)")
    parser.add_argument("--skip_step_3", action="store_true",
                        help="If set, will skip ACDC Step 3 (Simulated Scene Generation)")
    parser.add_argument("--step_1_output_path", type=str, default=None,
                        help="output path from Step 1 to use. Only necessary if --skip_step_1 is set and --skip_step_2 is not set.")
    parser.add_argument("--step_2_output_path", type=str, default=None,
                        help="output path from Step 2 to use. Only necessary if --skip_step_2 is set and --skip_step_3 is not set.")

    args = parser.parse_args()
    main(args)

