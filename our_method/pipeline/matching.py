
"""
Digital Cousin Matcher for ACDC pipeline (Step 2).
Finds and matches digital cousin candidates from a dataset using foundation models (CLIP, GPT, DINOv2).
"""


# --- Standard library imports ---
import os
import json
import re
import warnings
from pathlib import Path

# --- Third-party imports ---
import torch
import numpy as np
from PIL import Image
import cv2
import faiss
from loguru import logger as log
from torchvision.ops.boxes import _box_xyxy_to_cxcywh

# --- Project imports ---
import our_method
from groundingdino.util.inference import load_image
from our_method.models.clip import CLIPEncoder
from our_method.models.gpt import GPT
from our_method.utils.processing_utils import prepare_output_dir, NumpyTorchEncoder, compute_bbox_from_mask
from our_method.utils.dataset_utils import (
    get_all_dataset_categories,
    get_all_articulated_categories,
    extract_info_from_model_snapshot,
    ARTICULATION_INFO,
    ARTICULATION_VALID_ANGLES,
)



DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}



class DigitalCousinMatcher:
    """
    Step 2 in the ACDC pipeline: Digital Cousin Matching.

    Given Step 1 output JSON from RealWorldExtractor, selects digital cousins (category, model, pose)
    for each detected object and saves them into step_2_output_info.json.
    """


    def __init__(self, feature_matcher, gpt=None, verbose: bool = False):
        """
        Args:
            feature_matcher: Feature matcher instance for visual retrieval.
            gpt: (Optional) GPT instance to use. If None, must be set before use.
            verbose: If True, enables verbose logging.
        """
        self.fm = feature_matcher
        self.fm.eval()
        self.verbose = verbose
        self.device = self.fm.device
        self.gpt = gpt

    def __call__(
        self,
        step_1_output_path,
        top_k_categories=3,
        top_k_models=8,
        top_k_poses=3,
        n_digital_cousins=3,
        n_cousins_reselect_cand=3,
        remove_background=False,
        gpt_select_cousins=True,
        n_cousins_link_count_threshold=3,
        save_dir=None,
        start_at_name=None,
    ):
        """
        Run the digital cousin matching pipeline.

        1. Use CLIP to find top-K dataset categories per object.
        2. For each object, use the feature matcher + GPT to select model and pose.
        """
        self.step_1_output_path = step_1_output_path
        self.top_k_categories = top_k_categories
        self.top_k_models = top_k_models
        self.top_k_poses = top_k_poses
        self.n_digital_cousins = n_digital_cousins
        self.n_cousins_reselect_cand = n_cousins_reselect_cand
        self.remove_background = remove_background
        self.gpt_select_cousins = gpt_select_cousins
        self.n_cousins_link_count_threshold = n_cousins_link_count_threshold
        self.start_at_name = start_at_name

        assert self.n_digital_cousins <= self.top_k_models, (
            f"n_digital_cousins ({self.n_digital_cousins}) cannot be greater than "
            f"top_k_models ({self.top_k_models})!"
        )

        self.save_dir = prepare_output_dir(self.step_1_output_path, save_dir, "step_2_output")

        if self.verbose:
            log.info(f"Computing digital cousins given output {self.step_1_output_path}...")
            log.debug("Sub-Step 1. Use CLIP embeddings to find top-K categories per object")

        self._load_step1_outputs()
        self._compute_topk_categories()

        if self.gpt is None:
            raise ValueError("gpt must be provided!")

        if self.verbose:
            log.debug("Sub-Step 2. Select digital cousins using encoder features + GPT")

        success = self._match_all_objects()

        if not success:
            return False, None

        step_2_output_path = self._save_step2_outputs()
        if self.verbose:
            log.info(f"Saved Step 2 Output information to {step_2_output_path}")
            log.success("Completed Digital Cousin Matching!")

        return True, step_2_output_path


    def _extract_first_int(self, text: str | None):
        """Extract the first integer from a string, or None if not found."""
        if text is None:
            return None
        match = re.search(r"\b\d+\b", text)
        return int(match.group()) if match else None

    def _load_step1_outputs(self):
        with open(self.step_1_output_path, "r") as f:
            self.step_1_output_info = json.load(f)
        with open(self.step_1_output_info["detected_categories"], "r") as f:
            self.detected_categories_info = json.load(f)

        self.input_rgb_path = self.step_1_output_info["input_rgb"]
        self.names = self.detected_categories_info["names"]
        self.phrases = self.detected_categories_info["phrases"]
        self.phrases_recaptioned = self.detected_categories_info["phrases_recaptioned"]
        self.segmentation_dir = self.detected_categories_info["segmentation_dir"]
        self.articulation_counts = {
            int(k): v for k, v in self.detected_categories_info["articulation_counts"].items()
        }
        self.boxes = torch.tensor(self.detected_categories_info["boxes"])
        self.logits = torch.tensor(self.detected_categories_info["logits"])





    def _compute_topk_categories(self):
        all_categories = list(
            get_all_dataset_categories(
                do_not_include_categories=DO_NOT_MATCH_CATEGORIES,
                replace_underscores=True,
            )
        )
        all_articulated_categories = list(
            get_all_articulated_categories(
                do_not_include_categories=DO_NOT_MATCH_CATEGORIES,
                replace_underscores=True,
            )
        )

        articulation_indexes = list(self.articulation_counts.keys())
        non_articulation_indexes = [
            idx for idx in range(len(self.phrases_recaptioned))
            if idx not in self.articulation_counts
        ]

        clip = CLIPEncoder(backbone_name="ViT-B/32", device=self.device)
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(clip.embedding_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        selected_categories = {}

        for obj_indexes, categories in zip(
            (non_articulation_indexes, articulation_indexes),
            (all_categories, all_articulated_categories),
        ):
            obj_phrases = [self.phrases_recaptioned[idx] for idx in obj_indexes]
            if len(obj_phrases) == 0:
                continue

            if self.verbose:
                log.info(f"Computing top-{self.top_k_categories} for phrases: {obj_phrases}")

            text_features = clip.get_text_features(text=categories)
            cand_text_features = clip.get_text_features(text=obj_phrases)
            gpu_index_flat.reset()
            gpu_index_flat.add(text_features)
            _dists, idxs = gpu_index_flat.search(cand_text_features, self.top_k_categories)

            for obj_idx, topk_idxs in zip(obj_indexes, idxs):
                selected_categories[self.names[obj_idx]] = [
                    categories[topk_idx] for topk_idx in topk_idxs
                ]

        topk_categories_info = {"topk_categories": selected_categories}
        if self.verbose:
            log.info(f"topk_categories_info: {topk_categories_info}")
        topk_categories_path = f"{self.save_dir}/topk_categories.json"
        with open(topk_categories_path, "w+") as f:
            json.dump(topk_categories_info, f, indent=4)

        del res
        del clip

        self.selected_categories = selected_categories



    def _match_all_objects(self) -> bool:
        should_start = self.start_at_name is None
        n_instances = len(self.names)

        for instance_idx, (box, logit, phrase, name) in enumerate(
            zip(self.boxes, self.logits, self.phrases, self.names)
        ):
            if not should_start:
                if self.start_at_name == name:
                    should_start = True
                else:
                    continue
            
            success = self._process_single_object(
                instance_idx=instance_idx,
                n_instances=n_instances,
                name=name,
                box=box,
                logit=logit,
                phrase=phrase,
            )
            if not success:
                return False

        return True



    def _process_single_object(
        self,
        instance_idx: int,
        n_instances: int,
        name: str,
        box,
        logit,
        phrase: str,
    ) -> bool:
        """
        End-to-end processing for a single object:
        1) Select model candidates.
        2) Select K digital cousins.
        3) For each cousin, select pose.
        """
        og_categories = self.selected_categories[name]
        obj_save_dir = f"{self.save_dir}/{name}"
        topk_model_candidates_dir = f"{obj_save_dir}/top_k_model_candidates"
        topk_pose_candidates_dir = f"{obj_save_dir}/top_k_pose_candidates"
        cousin_visualization_dir = f"{obj_save_dir}/cousin_visualization"
        obj_mask_fpath = f"{self.segmentation_dir}/{name}_nonprojected_mask.png"
        mask = np.array(Image.open(obj_mask_fpath))
        is_articulated = instance_idx in self.articulation_counts

        Path(cousin_visualization_dir).mkdir(parents=True, exist_ok=True)
        Path(topk_model_candidates_dir).mkdir(parents=True, exist_ok=True)
        Path(topk_pose_candidates_dir).mkdir(parents=True, exist_ok=True)

        category = phrase.replace(" ", "_")

        if self.verbose:
            log.info("-----------------")
            log.info(
                f"[Object {instance_idx + 1} / {n_instances}] "
                f"Finding for object {name}, category: {category}..."
            )

        obj_masks = mask.reshape(1, 1, *mask.shape)
        bboxes = _box_xyxy_to_cxcywh(
            torch.tensor(compute_bbox_from_mask(obj_mask_fpath))
        ).unsqueeze(dim=0)

        cousin_results = {"articulated": is_articulated, "cousins": []}
        selected_models = set()
        current_candidates = None

        for i in range(self.n_digital_cousins):
            log.info(f"Selecting digital cousin {i+1} / {self.n_digital_cousins}")

            if i % self.n_cousins_reselect_cand == 0 or i >= self.n_cousins_link_count_threshold:
                current_candidates = self._select_model_candidates_for_iteration(
                    iter_idx=i,
                    name=name,
                    category=category,
                    og_categories=og_categories,
                    is_articulated=is_articulated,
                    instance_idx=instance_idx,
                    selected_models=selected_models,
                    obj_masks=obj_masks,
                    bboxes=bboxes,
                    logit=logit,
                    phrase=phrase,
                    topk_model_candidates_dir=topk_model_candidates_dir,
                )

            if not current_candidates:
                raise ValueError(
                    f"Not enough candidates to choose digital cousins for {name}!"
                )

            nn_model_index = self._select_final_model_index(
                iter_idx=i,
                name=name,
                is_articulated=is_articulated,
                current_candidates=current_candidates,
                topk_model_candidates_dir=topk_model_candidates_dir,
                instance_idx=instance_idx,
            )
            if nn_model_index is None:
                return False

            candidate_model = current_candidates[nn_model_index]
            selected_models.add(candidate_model.split("/")[-1])

            og_category = candidate_model.split("/objects/")[-1].split("/snapshot/")[0]
            og_model = candidate_model.split(".")[0].split(f"{og_category}_")[-1]
            cousin_topk_pose_candidates_dir = f"{topk_pose_candidates_dir}/cousin{i}"

            pose_results = self._select_pose_candidates_and_run_nn(
                name=name,
                category=category,
                og_category=og_category,
                og_model=og_model,
                cousin_topk_pose_candidates_dir=cousin_topk_pose_candidates_dir,
                bboxes=bboxes,
                logit=logit,
                phrase=phrase,
                obj_masks=obj_masks,
            )
            
            nn_pose_index = self._select_final_pose_index(
                iter_idx=i,
                name=name,
                pose_results=pose_results,
                instance_idx=instance_idx,
                topk_model_candidates_dir=topk_model_candidates_dir,
            )
            if nn_pose_index is None:
                return False

            snapshot_path = pose_results["candidates"][nn_pose_index]
            _, _, ori_offset, z_angle = extract_info_from_model_snapshot(snapshot_path)
            cousin_info = {
                "category": og_category,
                "model": og_model,
                "ori_offset": ori_offset,
                "z_angle": z_angle,
                "snapshot": snapshot_path,
            }
            cousin_results["cousins"].append(cousin_info)

            image_source, _image = load_image(self.input_rgb_path)
            ref_img_vis = cv2.resize(image_source, (640, 480))
            nn_img = np.array(Image.open(snapshot_path).convert("RGB"))
            imgs = [ref_img_vis, cv2.resize(nn_img, (640, 480))]
            concat_img = np.concatenate(imgs, axis=1)
            Image.fromarray(concat_img).save(
                f"{cousin_visualization_dir}/cousin{i}_visualization.png"
            )

            current_candidates.pop(nn_model_index)

        
        obj_cousin_results_path = f"{obj_save_dir}/cousin_results.json"
        with open(obj_cousin_results_path, "w+") as f:
            json.dump(cousin_results, f, indent=4, cls=NumpyTorchEncoder)

        if self.verbose:
            log.info("-----------------\n")

        return True



    def _select_model_candidates_for_iteration(
        self,
        iter_idx: int,
        name: str,
        category: str,
        og_categories: list[str],
        is_articulated: bool,
        instance_idx: int,
        selected_models: set,
        obj_masks: np.ndarray,
        bboxes: torch.Tensor,
        logit,
        phrase: str,
        topk_model_candidates_dir: str,
    ) -> list[str]:
        if self.verbose:
            log.info(f"Reselecting candidates using {self.fm.encoder_name}...")

        candidate_imgs_fdirs = [
            f"{our_method.ASSET_DIR}/objects/{og_category.replace(' ', '_')}/snapshot"
            for og_category in og_categories
        ]

        # Build candidate image paths
        if is_articulated:
            if self.verbose:
                log.info(
                    "Articulated object being matched, filtering candidate models "
                    "to valid number of doors / drawers"
                )

            input_n_doors, input_n_drawers = self.articulation_counts[instance_idx]
            input_n_doors_drawers = input_n_doors + input_n_drawers

            raw_all_cand_img_fpaths = list(
                sorted(
                    f"{candidate_imgs_fdir}/{model}"
                    for candidate_imgs_fdir in candidate_imgs_fdirs
                    for model in os.listdir(candidate_imgs_fdir)
                    if model not in selected_models
                )
            )

            if iter_idx < self.n_cousins_link_count_threshold:
                candidates_door_drawer_count = {}
                for candidate_img_fpath in raw_all_cand_img_fpaths:
                    filename = candidate_img_fpath.split("/")[-1].split(".")[0]
                    cand_category = candidate_img_fpath.split("/")[-3]
                    model = filename.split("_")[-1]
                    n_doors = int(ARTICULATION_INFO[cand_category][model][0])
                    n_drawers = int(ARTICULATION_INFO[cand_category][model][1])
                    candidates_door_drawer_count[candidate_img_fpath] = [n_doors, n_drawers]

                if input_n_doors_drawers <= 2:
                    tolerance = 0
                elif input_n_doors_drawers <= 4:
                    tolerance = 1
                elif input_n_doors_drawers <= 6:
                    tolerance = 2
                elif input_n_doors_drawers <= 8:
                    tolerance = 3
                else:
                    tolerance = 4

                candidate_imgs = []
                while len(candidate_imgs) == 0:
                    if tolerance > 4:
                        raise ValueError(
                            f"Failed to find valid candidates within reasonable "
                            f"tolerance for articulated object {name}!"
                        )
                    candidate_imgs = [
                        c_fpath
                        for c_fpath, (n_doors, n_drawers) in candidates_door_drawer_count.items()
                        if (abs(n_doors - input_n_doors) + abs(n_drawers - input_n_drawers))
                        <= tolerance
                    ]
                    tolerance += 1
            else:
                candidate_imgs = raw_all_cand_img_fpaths
        else:
            candidate_imgs = list(
                sorted(
                    f"{candidate_imgs_fdir}/{model}"
                    for candidate_imgs_fdir in candidate_imgs_fdirs
                    for model in os.listdir(candidate_imgs_fdir)
                    if model not in selected_models
                )
            )

        log.info(f"len(candidate_imgs): {len(candidate_imgs)}")

        # Run feature matcher to select top-K models
        if self.verbose:
            log.info(
                f"Selecting Top-{self.top_k_models} nearest models using {self.fm.encoder_name}..."
            )

        max_candidates = 120
        last_prefix = None

        if len(candidate_imgs) < max_candidates:
            last_prefix = f"{name}_iter{iter_idx}"
            model_results = self.fm.find_nearest_neighbor_candidates(
                input_category=category,
                input_img_fpath=self.input_rgb_path,
                candidate_imgs_fdirs=None,
                candidate_imgs=candidate_imgs,
                candidate_filter=None,
                n_candidates=self.top_k_models,
                save_dir=topk_model_candidates_dir,
                visualize_resolution=(640, 480),
                boxes=bboxes,
                logits=logit.unsqueeze(dim=0),
                phrases=[phrase],
                obj_masks=obj_masks,
                save_prefix=last_prefix,
                remove_background=self.remove_background,
            )
        else:
            num_batches = len(candidate_imgs) // max_candidates + 1
            log.info(f"Too many candidates, splitting into {num_batches} batches...")
            model_results_list = []
            for batch_idx in range(num_batches):
                log.info(f"Batch {batch_idx + 1}/{num_batches}")
                candidate_imgs_temp = candidate_imgs[
                    batch_idx * max_candidates : (batch_idx + 1) * max_candidates
                ]
                log.info(f"len(candidate_imgs_temp): {len(candidate_imgs_temp)}")
                _batch_results = self.fm.find_nearest_neighbor_candidates(
                    input_category=category,
                    input_img_fpath=self.input_rgb_path,
                    candidate_imgs_fdirs=None,
                    candidate_imgs=candidate_imgs_temp,
                    candidate_filter=None,
                    n_candidates=self.top_k_models,
                    save_dir=topk_model_candidates_dir,
                    visualize_resolution=(640, 480),
                    boxes=bboxes,
                    logits=logit.unsqueeze(dim=0),
                    phrases=[phrase],
                    obj_masks=obj_masks,
                    save_prefix=f"{name}_iter{batch_idx}",
                    remove_background=self.remove_background,
                )
                model_results_list.extend(_batch_results["candidates"])

            log.info(f"len(model_results_list): {len(model_results_list)}")
            last_prefix = f"{name}_iter{num_batches}"
            model_results = self.fm.find_nearest_neighbor_candidates(
                input_category=category,
                input_img_fpath=self.input_rgb_path,
                candidate_imgs_fdirs=None,
                candidate_imgs=model_results_list,
                candidate_filter=None,
                n_candidates=self.top_k_models,
                save_dir=topk_model_candidates_dir,
                visualize_resolution=(640, 480),
                boxes=bboxes,
                logits=logit.unsqueeze(dim=0),
                phrases=[phrase],
                obj_masks=obj_masks,
                save_prefix=last_prefix,
                remove_background=self.remove_background,
            )

        # Normalize visualization file names
        src_bbox = f"{topk_model_candidates_dir}/{last_prefix}_annotated_bboxes.png"
        src_mask = f"{topk_model_candidates_dir}/{last_prefix}_mask.png"
        dst_bbox = f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png"
        dst_mask = f"{topk_model_candidates_dir}/{name}_mask.png"

        if os.path.exists(src_bbox):
            os.replace(src_bbox, dst_bbox)
        if os.path.exists(src_mask):
            os.replace(src_mask, dst_mask)

        return model_results["candidates"]

    def _select_final_model_index(
        self,
        iter_idx: int,
        name: str,
        is_articulated: bool,
        current_candidates: list[str],
        topk_model_candidates_dir: str,
        instance_idx: int,
    ) -> int | None:
        if not self.gpt_select_cousins:
            if self.verbose:
                log.info(f"Selecting cousin #{iter_idx} final model using DINOv2...")
            return 0

        if self.verbose:
            log.info(f"Selecting cousin #{iter_idx} final model using GPT...")

        bbox_img_path = f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png"

        if is_articulated:
            payload = self.gpt.payload_articulated_nearest_neighbor(
                caption=self.phrases_recaptioned[instance_idx],
                img_path=self.input_rgb_path,
                bbox_img_path=bbox_img_path,
                candidates_fpaths=current_candidates,
            )
        else:
            payload = self.gpt.payload_nearest_neighbor(
                caption=self.phrases_recaptioned[instance_idx],
                img_path=self.input_rgb_path,
                bbox_img_path=bbox_img_path,
                candidates_fpaths=current_candidates,
                nonproject_obj_img_path=f"{self.segmentation_dir}/{name}_nonprojected.png",
            )

        resp = self.gpt(payload=payload, verbose=self.verbose)
        if resp is None:
            log.info(resp)
            return None

        base_index = self._extract_first_int(resp)
        if base_index is None:
            return None

        nn_model_index = base_index - 1
        if nn_model_index < 0 or nn_model_index >= len(current_candidates):
            nn_model_index = 0

        return nn_model_index



    def _select_pose_candidates_and_run_nn(
        self,
        name: str,
        category: str,
        og_category: str,
        og_model: str,
        cousin_topk_pose_candidates_dir: str,
        bboxes: torch.Tensor,
        logit,
        phrase: str,
        obj_masks: np.ndarray,
    ):
        if self.verbose:
            log.info(
                f"Selecting Top-{self.top_k_poses} nearest poses using {self.fm.encoder_name}..."
            )

        start_idx, end_idx = ARTICULATION_VALID_ANGLES.get(og_category, {}).get(
            og_model, [0, 99]
        )
        candidate_imgs = [
            f"{our_method.ASSET_DIR}/objects/{og_category}/model/{og_model}/{og_model}_{rot_idx}.png"
            for rot_idx in range(start_idx, end_idx + 1)
        ]

        pose_results = self.fm.find_nearest_neighbor_candidates(
            input_category=category,
            input_img_fpath=self.input_rgb_path,
            candidate_imgs_fdirs=None,
            candidate_imgs=candidate_imgs,
            n_candidates=self.top_k_poses,
            save_dir=cousin_topk_pose_candidates_dir,
            visualize_resolution=(640, 480),
            boxes=bboxes,
            logits=logit.unsqueeze(dim=0),
            phrases=[phrase],
            obj_masks=obj_masks,
            save_prefix=name,
            remove_background=self.remove_background,
        )
        return pose_results

    def _select_final_pose_index(
        self,
        iter_idx: int,
        name: str,
        pose_results: dict,
        instance_idx: int,
        topk_model_candidates_dir: str,
    ) -> int | None:
        if self.verbose:
            log.info(f"Selecting cousin #{iter_idx} final pose using GPT...")

        bbox_img_path = f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png"
        payload = self.gpt.payload_nearest_neighbor_pose(
            caption=self.phrases_recaptioned[instance_idx],
            img_path=self.input_rgb_path,
            bbox_img_path=bbox_img_path,
            nonproject_obj_img_path=f"{self.segmentation_dir}/{name}_nonprojected.png",
            candidates_fpaths=pose_results["candidates"],
        )

        resp = self.gpt(payload=payload, verbose=self.verbose)
        if resp is None:
            return None

        idx = self._extract_first_int(resp)
        if idx is None:
            warnings.warn(
                f"Got invalid response! Valid options are pose indices, got: '{resp}'"
            )
            idx = 0

        if idx >= len(pose_results["candidates"]):
            idx = 0

        return idx



    def _save_step2_outputs(self) -> str:
        step_2_output_info = {
            "metadata": {
                "n_cousins": self.n_digital_cousins,
                "n_objects": len(self.names),
            },
            "objects": {},
        }

        for name in self.names:
            obj_cousin_results_path = f"{self.save_dir}/{name}/cousin_results.json"
            with open(obj_cousin_results_path, "r") as f:
                obj_cousin_results = json.load(f)
            step_2_output_info["objects"][name] = obj_cousin_results

        step_2_output_path = f"{self.save_dir}/step_2_output_info.json"
        with open(step_2_output_path, "w+") as f:
            json.dump(step_2_output_info, f, indent=4, cls=NumpyTorchEncoder)

        return step_2_output_path
