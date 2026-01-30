
# --- Standard library imports ---
import os
import json
import multiprocessing
from pathlib import Path
from copy import deepcopy

# --- Third-party imports ---
import torch
import numpy as np
from PIL import Image
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from skimage import morphology
from loguru import logger as log

# --- Project imports ---
from torchvision.ops.boxes import box_convert
from groundingdino.util.inference import load_image
import our_method.utils.transform_utils as T
from our_method.models.gpt import GPT
from our_method.models.perspective_fields import PerspectiveFields
from our_method.models.depth_anything_v2 import DepthAnythingV2
from our_method.models.unidepth_v2 import UniDepthV2Wrapper
from our_method.utils.processing_utils import (
    prepare_output_dir,
    create_polygon_from_vertices,
    NumpyTorchEncoder,
    filter_large_masks,
    unprocess_depth_linear,
    process_depth_linear,
    compute_point_cloud_from_depth,
    annotate,
    mask_intersection_area,
    mask_area,
    shrink_mask,
    denoise_obj_point_cloud,
    distance_to_plane,
    get_aabb_vertices,
    project_vertices_to_plane,
    get_possible_obj_on_wall,
)



class RealWorldExtractor:
    """
    Step 1 of the pipeline: From a single RGB image, segment objects and estimate depth.
    One call = one run. Intermediate results are stored on self.*.
    """


    FLOOR_CATEGORY = "floor"
    POLYGON_RELATIVE_INTERSECTION_THRESHOLD = 0.97
    POLYGON_RELATIVE_AREA_THRESHOLD = 0.9
    OBJ_MASK_INTERSECT_AREA_THRESHOLD = 0.8


    def __init__(self, feature_matcher, gpt=None, verbose: bool = False):
        self.fm = feature_matcher
        self.fm.eval()
        self.verbose = verbose
        self.device = self.fm.device
        self.gpt = gpt


    def __call__(
        self,
        input_path,
        input_depth_path,
        captions=None,
        camera_intrinsics_matrix=None,
        depth_image=None,
        depth_model="UniDepthV2",
        depth_max_limit=20.0,
        filter_backsplash=True,
        infer_mounting_type=True,
        infer_aligned_wall=True,
        save_dir=None,
        visualize=False,
    ):
        self.input_path = input_path
        self.input_depth_path = input_depth_path
        self.depth_model = depth_model
        self.depth_max_limit = depth_max_limit
        self.filter_backsplash = filter_backsplash
        self.infer_mounting_type = infer_mounting_type
        self.infer_aligned_wall = infer_aligned_wall
        self.camera_intrinsics_matrix = camera_intrinsics_matrix
        self.captions = captions
        self.visualize = visualize
        self.save_dir = prepare_output_dir(self.input_path, save_dir, "step_1_output")
        self._load_and_resize_image()

        if self.verbose:
            log.info(f"Extracting real-world info from image {self.input_path}...")
        self._infer_captions_if_needed()
        if self.captions is None:
            return False, None

        self._segment_floor_wall()
        if self.segmentation_dir is None:
            return False, None

        self._estimate_depth()
        self._estimate_intrinsics()
        self._compute_floor_plane_and_pc()
        self._estimate_wall_planes()

        self._parse_detected_objects()
        if self.verbose:
            log.info(f"Detected parsed unique object categories: {self.detected_objs}")

        self._predict_boxes_with_gsam()
        self._postprocess_boxes_and_masks()

        self._assign_object_names()
        if self.verbose:
            log.info(f"Remaining phrases after pruning: {self.phrases}")
            log.info(f"Object names: {self.names}")
            log.debug("Sub-Step 6. Re-prompt GPT to align original caption with GSAMv2's caption")

        self._compute_z_direction_and_tilt()

        self._refine_objects_with_gpt()
        if self.phrases_recaptioned is None:
            return False, None

        if self.visualize:
            self._visualize_objects_and_walls()

        self._infer_mounting_and_alignment()
        if self.mount_info is None:
            return False, None

        if self.verbose:
            log.info(f"Initial phrases: {self.phrases}")
            log.info(f"Final phrases after recaption: {self.phrases_recaptioned}")
            log.info("\nUpdating articulated object door / drawer count...\n")

        self._update_articulation_counts()
        if self.articulation_counts is None:
            return False, None

        self._save_outputs()

        if self.verbose:
            log.info(f"Saved extracted information to {self.detected_categories_path}")
            log.success("Completed Real World Extraction!")

        return True, self.step_1_output_path


    def _run_o3d_vis(self, fn):
        p = multiprocessing.Process(target=fn)
        p.start()
        p.join()

    def _prepare_output_dir(self, input_path, save_dir):
        if save_dir is None:
            save_dir = os.path.dirname(input_path)
        save_dir = os.path.join(save_dir, "step_1_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return save_dir

    def _load_and_resize_image(self, target_long_edge=1600):
        raw_rgb = Image.open(self.input_path).convert("RGB")
        raw_width, raw_height = raw_rgb.size

        if raw_width > raw_height:
            new_width = target_long_edge
            new_height = int((target_long_edge / raw_width) * raw_height)
            scale_factor = target_long_edge / raw_width
        else:
            new_height = target_long_edge
            new_width = int((target_long_edge / raw_height) * raw_width)
            scale_factor = target_long_edge / raw_height

        resized_rgb = raw_rgb.resize((new_width, new_height), Image.BICUBIC)

        name_seg = self.input_path.split(".")
        name_seg[-2] += "_resize"
        self.input_path = ".".join(name_seg)
        resized_rgb.save(self.input_path)

        if self.verbose:
            log.info(
                f"Input image ({raw_width}, {raw_height}) resized to "
                f"({new_width}, {new_height})"
            )
            log.info(f"Saved resized image to {self.input_path}")

        self.rgb = np.array(resized_rgb)
        self.new_width = new_width
        self.new_height = new_height
        self.scale_factor = scale_factor

    def _infer_captions_if_needed(self):
        if self.captions is not None:
            return

        if self.verbose:
            log.debug("Sub-Step 0. Use GPT to infer objects")

        payload = self.gpt.payload_get_object_caption(img_path=self.input_path)
        resp = self.gpt(payload=payload, verbose=self.verbose)
        if resp is None:
            self.captions = None
            return

        self.captions = set(self.gpt.extract_captions(gpt_text=resp))
        if self.verbose:
            log.info(f"Detected raw object captions: {self.captions}")

    def _segment_floor_wall(self):
        if self.verbose:
            log.debug("Sub-Step 1. Compute segmentation masks for floor, backsplash, walls")

        segmentation_dir = f"{self.save_dir}/segmented_objects"
        Path(segmentation_dir).mkdir(parents=True, exist_ok=True)

        _, floor_mask_paths = self.fm.compute_segmentation_mask(
            input_category="floor",
            input_img_fpath=self.input_path,
            save_dir=f"{segmentation_dir}/floor",
        )
        _, backsplash_mask_paths = self.fm.compute_segmentation_mask(
            input_category="backsplash",
            input_img_fpath=self.input_path,
            save_dir=f"{segmentation_dir}/backsplash",
        )
        _, wall_mask_paths = self.fm.compute_segmentation_mask(
            input_category="wall",
            input_img_fpath=self.input_path,
            save_dir=f"{segmentation_dir}/wall",
            multi_results=True,
        )

        assert len(floor_mask_paths) == 1, "Got more than one floor segmentation!"
        floor_mask_path = floor_mask_paths[0]

        if self.filter_backsplash:
            filtered_wall_mask_paths = []
            for cand_mask_path in backsplash_mask_paths + wall_mask_paths:
                color_mask_path = cand_mask_path.replace("_mask.png", "_nonprojected.png")
                mask = np.array(Image.open(cand_mask_path))
                color_mask_img = self.rgb * np.expand_dims(mask, axis=-1)
                Image.fromarray(color_mask_img).save(color_mask_path)

                payload = self.gpt.payload_filter_wall(
                    img_path=self.input_path,
                    candidate_fpath=color_mask_path,
                )
                resp = self.gpt(payload=payload, verbose=self.verbose)
                if resp is None:
                    self.segmentation_dir = None
                    self.floor_mask_path = None
                    self.raw_wall_mask_paths = None
                    return

                if "y" in resp:
                    filtered_wall_mask_paths.append(cand_mask_path)
                elif "n" not in resp:
                    raise ValueError(
                        f"Got invalid response! Valid options are: [y, n], got: {resp}"
                    )
        else:
            filtered_wall_mask_paths = backsplash_mask_paths + wall_mask_paths

        raw_wall_mask_paths = filter_large_masks(filtered_wall_mask_paths)
        color_wall_mask_paths = [
            p.replace("_mask.png", "_nonprojected.png") for p in raw_wall_mask_paths
        ]

        if self.verbose:
            log.info(f"Filtered Wall/Backsplash Masks: {color_wall_mask_paths}")

        self.segmentation_dir = segmentation_dir
        self.floor_mask_path = floor_mask_path
        self.raw_wall_mask_paths = raw_wall_mask_paths

    def _estimate_intrinsics(self):
        if self.camera_intrinsics_matrix is None:
            if self.verbose:
                log.info("No K intrinsics matrix given, estimating...")
            intrinsics_estimator = PerspectiveFields(device=self.device)
            intrinsics = intrinsics_estimator.estimate_camera_intrinsics(
                input_path=self.input_path
            )
            intrinsics_estimator.to("cpu")
            del intrinsics_estimator
            self.camera_intrinsics_matrix = np.array(intrinsics)
        else:
            K = self.camera_intrinsics_matrix.copy()
            K[0, 0] *= self.scale_factor
            K[1, 1] *= self.scale_factor
            K[0, 2] *= self.scale_factor
            K[1, 2] *= self.scale_factor
            self.camera_intrinsics_matrix = K
        
        if self.verbose:
            log.info(
                f"Camera intrinsics scaled by factor {self.scale_factor:.4f}, "
                f"K matrix:\n{self.camera_intrinsics_matrix}"
            )


    def _estimate_depth(self):
        if self.verbose:
            log.debug(f"Sub-Step 2. Run {self.depth_model} to extract synthetic depth map")
            log.info("Estimating depth map...")

        depth_path = f"{self.save_dir}/step_1_depth.png"
        depth_limits = np.array([0, self.depth_max_limit])

        if self.input_depth_path is not None and os.path.exists(self.input_depth_path):
            depth = np.load(self.input_depth_path)
            output_shape = (self.new_height, self.new_width)
            depth_image = process_depth_linear(
                depth=depth,
                in_limits=depth_limits,
                out_shape=output_shape,
            )
            Image.fromarray(depth_image).save(depth_path)
            if self.verbose:
                log.info(
                    f"Loaded depth image from {self.input_depth_path}, "
                    f"shape: {depth_image.shape}, dtype: {depth_image.dtype}"
                )
        else:
            if self.depth_model == "DepthAnythingV2":
                depth_estimator = DepthAnythingV2(device=self.device)
                depth_image = depth_estimator.estimate_depth_linear(
                    input_path=self.input_path,
                    output_path=depth_path,
                    depth_limits=depth_limits,
                )
                depth_estimator.to("cpu")
                del depth_estimator
            elif self.depth_model == "UniDepthV2":
                depth_estimator = UniDepthV2Wrapper(device=self.device)
                self.camera_intrinsics_matrix, depth_image = depth_estimator.estimate_depth_and_intrinsic_linear(
                    input_path=self.input_path,
                    output_path=depth_path,
                    depth_limits=depth_limits,
                )
                self.scale_factor = 1.0
                depth_estimator.to("cpu")
                del depth_estimator
            else:
                log.warning(f"Unknown depth model {self.depth_model}, defaulting to UniDepthV2")
                depth_estimator = UniDepthV2Wrapper(device=self.device)
                depth_image = depth_estimator.estimate_depth_and_intrinsic_linear(
                    input_path=self.input_path,
                    output_path=depth_path,
                    depth_limits=depth_limits,
                )
                depth_estimator.to("cpu")
                del depth_estimator

        if self.verbose:
            log.info(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")

        self.depth_path = depth_path
        self.depth_limits = depth_limits

    def _compute_floor_plane_and_pc(self):
        if self.verbose:
            log.debug("Sub-Step 3. Compute z-direction given segmented floor plane")

        floor_mask = np.array(Image.open(self.floor_mask_path))
        depth = unprocess_depth_linear(
            np.array(Image.open(self.depth_path)),
            out_limits=self.depth_limits,
        )
        pc = compute_point_cloud_from_depth(depth=depth, K=self.camera_intrinsics_matrix)

        if self.visualize:
        # if True:
            def vis():
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(self.rgb.reshape(-1, 3) / 255.0)
                o3d.visualization.draw_geometries([pcd])

            self._run_o3d_vis(vis)
        
        floor_idx = floor_mask.flatten().nonzero()[0]

        if len(floor_idx) == 0:
            if self.verbose:
                log.error("Floor mask is empty! No floor points detected.")
            raise ValueError("Floor segmentation failed: no floor points found")

        pc_flat = pc.reshape(-1, 3)
        rgb_flat = self.rgb.reshape(-1, 3)

        floor_idx = floor_mask.flatten().nonzero()[0]
        pc_floor = pc_flat[floor_idx]
        rgb_floor = rgb_flat[floor_idx]


        pcd = o3d.geometry.PointCloud()
        pc_floor_mean = np.mean(pc_floor, axis=0)
        pcd.points = o3d.utility.Vector3dVector(pc_floor - pc_floor_mean.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_floor / 255.0)

        # Visualize floor point cloud if visualize is enabled
        if self.visualize:
        # if True:
            def vis_floor():
                o3d.visualization.draw_geometries([pcd])

            self._run_o3d_vis(vis_floor)
        pcd.points = o3d.utility.Vector3dVector(pc_floor - pc_floor_mean.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_floor / 255.0)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=10,
        )
        a, b, c, d = plane_model
        z_dir_plane = np.array([a, b, c])

        if self.verbose:
            log.info(
                f"Estimated floor plane equation: "
                f"{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0"
            )

        inlier_cloud = pcd.select_by_index(inliers)
        pc_floor = np.asarray(inlier_cloud.points)
        origin_pos = pc_floor[int(len(pc_floor) // 2)] + pc_floor_mean

        if self.verbose:
            log.info(f"Selected origin_pos: {origin_pos}")

        self.pc = pc
        self.floor_mask = floor_mask
        self.z_dir_plane = z_dir_plane
        self.pc_floor = pc_floor
        self.pc_floor_mean = pc_floor_mean
        self.origin_pos = origin_pos
        self.inlier_cloud = inlier_cloud

    def _estimate_wall_planes(self):
        all_wall_mask_planes = {}
        for i, wall_mask_path in enumerate(self.raw_wall_mask_paths):
            wall_mask = Image.open(wall_mask_path)
            shrunk_wall_mask = shrink_mask(np.array(wall_mask), iterations=2)
            pc_wall = self.pc.reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]]

            pcd_wall = o3d.geometry.PointCloud()
            pcd_wall.points = o3d.utility.Vector3dVector(pc_wall)
            pcd_wall.colors = o3d.utility.Vector3dVector(
                self.rgb.reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]] / 255.0
            )
            pcd_wall = pcd_wall.uniform_down_sample(every_k_points=5)
            pcd_wall, _ = pcd_wall.remove_statistical_outlier(
                nb_neighbors=16,
                std_ratio=1.5,
            )
            pc_wall = np.asarray(pcd_wall.points)

            plane_model, _ = pcd_wall.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=1000,
            )
            a, b, c, d = plane_model
            wall_normal_vec = np.array([a, b, c])
            wall_normal_vec = wall_normal_vec / np.linalg.norm(wall_normal_vec)
            start_point = np.median(pc_wall, axis=0)
            wall_normal_vec = -np.sign(np.dot(wall_normal_vec, start_point)) * wall_normal_vec
            all_wall_mask_planes[wall_mask_path] = {
                "normal": wall_normal_vec,
                "point": start_point,
            }

            if self.verbose:
                log.info(
                    f"Estimated wall {i}'s plane equation: "
                    f"{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0"
                )

        self.all_wall_mask_planes = all_wall_mask_planes

    def _parse_detected_objects(self):
        detected_objs = {}
        for raw_caption in self.captions:
            obj_category = raw_caption.split("(")[0]
            if obj_category in detected_objs:
                continue
            if "(" in raw_caption and ")" in raw_caption:
                n_doors = int(raw_caption.split(" door")[0][-1]) if "door" in raw_caption else 0
                n_drawers = int(raw_caption.split(" drawer")[0][-1]) if "drawer" in raw_caption else 0
                detected_objs[obj_category] = (n_doors, n_drawers)
            else:
                detected_objs[obj_category] = None
        self.detected_objs = detected_objs

    def _predict_boxes_with_gsam(self):
        image_source, image = load_image(self.input_path)

        if self.verbose:
            log.info("Predicting input image bounding boxes using GroundingDINO...")

        boxes, logits, phrases = self.fm.gsam.predict_boxes(
            image,
            ". ".join(self.detected_objs.keys()),
        )

        self.image_source = image_source
        self.boxes = boxes
        self.logits = logits
        self.phrases = list(phrases)

    def _postprocess_boxes_and_masks(self):
        if self.verbose:
            log.debug(
                "Sub-Step 5. Post-process masks to remove noise and "
                "differentiate instances within the same mask"
            )
            log.info("Pruning redundant bounding boxes...")

        boxes_xyxy = box_convert(
            boxes=self.boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).numpy()

        idxs_to_remove = set()
        susp_small_large_box_idxs = set()

        for i, (box_a, phrase_a) in enumerate(zip(boxes_xyxy, self.phrases)):
            lower_a_x, lower_a_y, upper_a_x, upper_a_y = box_a
            polygon_a = create_polygon_from_vertices(
                [
                    (lower_a_x, lower_a_y),
                    (upper_a_x, lower_a_y),
                    (upper_a_x, upper_a_y),
                    (lower_a_x, upper_a_y),
                ]
            )

            for j, (box_b, phrase_b) in enumerate(zip(boxes_xyxy, self.phrases)):
                if i == j:
                    continue

                if phrase_a == self.FLOOR_CATEGORY:
                    idxs_to_remove.add(i)
                    continue

                if phrase_a in phrase_b or phrase_b in phrase_a:
                    lower_b_x, lower_b_y, upper_b_x, upper_b_y = box_b
                    polygon_b = create_polygon_from_vertices(
                        [
                            (lower_b_x, lower_b_y),
                            (upper_b_x, lower_b_y),
                            (upper_b_x, upper_b_y),
                            (lower_b_x, upper_b_y),
                        ]
                    )

                    if polygon_a.intersects(polygon_b):
                        intersect_area = polygon_a.intersection(polygon_b).area
                        if (
                            intersect_area / polygon_a.area
                            >= self.POLYGON_RELATIVE_INTERSECTION_THRESHOLD
                            or intersect_area / polygon_b.area
                            >= self.POLYGON_RELATIVE_INTERSECTION_THRESHOLD
                        ):
                            if polygon_a.area < polygon_b.area:
                                if (
                                    polygon_a.area
                                    < self.POLYGON_RELATIVE_AREA_THRESHOLD * polygon_b.area
                                ):
                                    susp_small_large_box_idxs.add((i, j))
                                else:
                                    idxs_to_remove.add(i)
                            else:
                                if (
                                    polygon_b.area
                                    < self.POLYGON_RELATIVE_AREA_THRESHOLD * polygon_a.area
                                ):
                                    susp_small_large_box_idxs.add((j, i))
                                else:
                                    idxs_to_remove.add(j)

        for i, (smaller_idx, larger_idx) in enumerate(susp_small_large_box_idxs):
            for j, (smaller_idx_2, larger_idx_2) in enumerate(susp_small_large_box_idxs):
                if i == j:
                    continue
                if larger_idx == larger_idx_2:
                    idxs_to_remove.add(larger_idx)
                    break

        all_masks = self.fm.gsam.predict_segmentation(
            self.image_source,
            self.boxes,
            multimask_output=True,
        )
        assert len(all_masks.shape) == 4, (
            "Expected masks to have shape 4 (N, num_masks, W, H), "
            f"instead got masks shape: {all_masks.shape}"
        )
        _, _, W, H = all_masks.shape

        masks = []
        for obj_all_mask in all_masks:
            mask_area_idx = [(np.sum(obj_all_mask[i]), i) for i in range(3)]
            mask_area_idx.sort()
            masks.append(
                obj_all_mask[mask_area_idx[0][1]] | obj_all_mask[mask_area_idx[1][1]]
            )

        for i, mask in enumerate(masks):
            masks[i] = morphology.remove_small_objects(
                mask,
                min_size=int(np.sqrt(W * H) / 10.0),
                connectivity=1,
            )

        for i, mask_a in enumerate(masks):
            if i in idxs_to_remove:
                continue
            for j, mask_b in enumerate(masks):
                if (i == j) or (j in idxs_to_remove):
                    continue
                inter_area = mask_intersection_area(mask_a, mask_b)
                min_area = min(mask_area(mask_a), mask_area(mask_b))
                if inter_area > self.OBJ_MASK_INTERSECT_AREA_THRESHOLD * min_area:
                    if mask_area(mask_a) > mask_area(mask_b):
                        idxs_to_remove.add(j)
                    else:
                        idxs_to_remove.add(i)
                        break

        for idx in sorted(idxs_to_remove, reverse=True):
            self.boxes = torch.cat((self.boxes[:idx], self.boxes[idx + 1:]))
            self.logits = torch.cat((self.logits[:idx], self.logits[idx + 1:]))
            del masks[idx]
            if self.verbose:
                log.info(f"remove {idx} {self.phrases[idx]}")
            del self.phrases[idx]

        self.masks = masks

    def _assign_object_names(self):
        category_counts = {}
        names = []
        for phrase in self.phrases:
            category = phrase.replace(" ", "_").replace("-", "_")
            idx = category_counts.get(category, 0)
            category_counts[category] = idx + 1
            names.append(f"{category}_{idx}")
        self.names = names

    def _compute_z_direction_and_tilt(self):
        obj_points = []
        for i, mask in enumerate(self.masks):
            if i > 2:
                break
            mask_idx = np.array(mask).flatten().nonzero()[0]
            pc_obj = self.pc.reshape(-1, 3)[mask_idx]
            pc_obj_median = np.median(pc_obj, axis=0)
            obj_points.append(pc_obj_median)

        floor2obj_vec = np.mean(obj_points, axis=0) - self.pc_floor_mean
        z_dir = np.sign(np.dot(floor2obj_vec, self.z_dir_plane)) * self.z_dir_plane

        if self.verbose:
            log.info(f"Estimated z-direction computed from floor point cloud: {z_dir}")

        assert abs(z_dir[0]) < 0.1, f"got tilted floor: {z_dir[0]}"

        if self.visualize:
            def vis():
                start_point = np.mean(self.pc_floor, axis=0)
                vector = z_dir / np.linalg.norm(z_dir)
                arrow_length = np.linalg.norm(vector)
                arrow_radius = 0.1
                arrow_cone_radius = 0.2
                arrow_cone_height = 0.5

                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, vector)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.dot(z_axis, vector))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                )

                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_radius,
                    cone_radius=arrow_cone_radius,
                    cylinder_height=arrow_length - arrow_cone_height,
                    cone_height=arrow_cone_height,
                )
                arrow.rotate(rotation_matrix, center=(0, 0, 0))
                arrow.translate(start_point)
                arrow_pcd = arrow.sample_points_poisson_disk(number_of_points=1000)
                o3d.visualization.draw_geometries(
                    [self.inlier_cloud, arrow_pcd],
                    point_show_normal=True,
                )

            self._run_o3d_vis(vis)

        tilt_angle = np.arctan2(z_dir[1], z_dir[2])
        tilt_mat = T.euler2mat([tilt_angle, 0, 0])
        self.pc = self.pc @ tilt_mat.T

        self.z_dir = z_dir
        self.tilt_mat = tilt_mat

    def _refine_objects_with_gpt(self):
        n_objects = len(self.names)
        phrases_recaptioned = deepcopy(self.phrases)
        clean_obj_pcd_boxes = []

        for i, (mask, box, phrase, name) in enumerate(
            zip(self.masks, self.boxes, self.phrases, self.names)
        ):
            if self.verbose:
                log.info("-----------------")
                log.info(f"Object {i + 1} / {n_objects}")

            original_obj_img = self.rgb * np.expand_dims(mask, axis=-1)
            nonprojected_img_path = f"{self.segmentation_dir}/{name}_nonprojected.png"
            mask_img_path = f"{self.segmentation_dir}/{name}_nonprojected_mask.png"
            annotated_bbox_img_path = f"{self.segmentation_dir}/{name}_annotated_bboxes.png"

            annotated_frame = annotate(
                image_source=self.image_source,
                boxes=box[None, :],
                phrases=["target"],
            )
            Image.fromarray(original_obj_img).save(nonprojected_img_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_img_path)
            cv2.imwrite(annotated_bbox_img_path, annotated_frame)

            payload = self.gpt.payload_select_object_from_list(
                img_path=self.input_path,
                obj_list=list(self.detected_objs.keys()),
                bbox_img_path=annotated_bbox_img_path,
                nonproject_obj_img_path=nonprojected_img_path,
            )
            if self.verbose:
                log.info("Inferring caption...")
            resp = self.gpt(payload=payload, verbose=self.verbose)
            if resp is None:
                self.phrases_recaptioned = None
                self.clean_obj_pcd_boxes = None
                return

            clean_caption = (
                resp.strip().strip('"').strip().strip('"').lower()
            )
            phrases_recaptioned[i] = clean_caption

            os.remove(annotated_bbox_img_path)
            annotated_frame = annotate(
                image_source=self.image_source,
                boxes=box[None, :],
                phrases=[clean_caption],
            )
            cv2.imwrite(annotated_bbox_img_path, annotated_frame)

            if self.verbose:
                log.info("Pruning masks...")

            mask_idx = np.array(mask).flatten().nonzero()[0]
            mask_pruned = np.zeros_like(mask).flatten()
            pc_obj = self.pc.reshape(-1, 3)[mask_idx]
            colors_obj = self.rgb.reshape(-1, 3)[mask_idx]

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pc_obj)
            pcd_obj.colors = o3d.utility.Vector3dVector(colors_obj / 255.0)

            clean_obj_pcd, valid_indices = denoise_obj_point_cloud(
                pcd_obj,
                visualize_result=self.visualize,
            )
            mask_pruned[mask_idx[valid_indices]] = 1.0
            mask_pruned = mask_pruned.reshape(mask.shape)
            obj_mask_pruned_fpath = (
                f"{self.segmentation_dir}/{name}_nonprojected_mask_pruned.png"
            )
            Image.fromarray(mask_pruned.astype(np.uint8) * 255).save(
                obj_mask_pruned_fpath
            )

            aabb = clean_obj_pcd.get_axis_aligned_bounding_box()
            clean_obj_pcd_boxes.append(aabb)

            if self.verbose:
                log.info(f"name: {self.names[i]}")
                log.info(f"phrase: {phrase}")
                log.info(f"clean_caption: {clean_caption}")
                log.info("-----------------\n")

        self.phrases_recaptioned = phrases_recaptioned
        self.clean_obj_pcd_boxes = clean_obj_pcd_boxes

    def _visualize_objects_and_walls(self):
        def vis():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pc.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(self.rgb.reshape(-1, 3) / 255.0)
            cmap = plt.get_cmap("tab20")

            for i, box in enumerate(self.clean_obj_pcd_boxes):
                color = cmap(i / len(self.names))[:3]
                box.color = color

            normal_arrows = []
            for wall_dict in self.all_wall_mask_planes.values():
                wall_normal_vec = self.tilt_mat @ np.array(wall_dict["normal"])
                wall_point = self.tilt_mat @ np.array(wall_dict["point"])
                arrow_radius = 0.05
                arrow_cone_radius = 0.1
                arrow_cone_height = 0.1
                cylinder_height = 0.2

                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, wall_normal_vec)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.dot(z_axis, wall_normal_vec))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                )

                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_radius,
                    cone_radius=arrow_cone_radius,
                    cylinder_height=cylinder_height,
                    cone_height=arrow_cone_height,
                )
                arrow.rotate(rotation_matrix, center=(0, 0, 0))
                arrow.translate(wall_point)
                arrow_pcd = arrow.sample_points_poisson_disk(number_of_points=1000)
                normal_arrows.append(arrow_pcd)

            o3d.visualization.draw_geometries(
                normal_arrows + self.clean_obj_pcd_boxes + [pcd]
            )

        self._run_o3d_vis(vis)

    def _infer_mounting_and_alignment(self):
        if self.verbose:
            log.info("\nPre-filtering possible mounting...\n")

        objs_possible_mounts = {name: [] for name in self.names}
        projected_cam_forward = np.array([0, 0, 1]) - np.dot(
            np.array([0, 0, 1]), self.z_dir
        ) * self.z_dir
        projected_cam_forward = projected_cam_forward / np.linalg.norm(
            projected_cam_forward
        )

        for wall_mask_path in self.raw_wall_mask_paths:
            wall_objs_info = []
            wall_normal_vec_original = self.all_wall_mask_planes[wall_mask_path]["normal"]
            projected_wall_normal = (
                np.array(wall_normal_vec_original)
                - np.dot(wall_normal_vec_original, self.z_dir) * self.z_dir
            )
            if np.linalg.norm(projected_wall_normal) < 1e-3:
                continue
            projected_wall_normal = projected_wall_normal / np.linalg.norm(
                projected_wall_normal
            )
            is_lateral_wall = (
                abs(np.dot(projected_cam_forward, projected_wall_normal)) < 0.707
            )

            for aabb, name in zip(self.clean_obj_pcd_boxes, self.names):
                wall_normal_vec = self.tilt_mat @ np.array(
                    self.all_wall_mask_planes[wall_mask_path]["normal"]
                )
                wall_point = self.tilt_mat @ np.array(
                    self.all_wall_mask_planes[wall_mask_path]["point"]
                )
                wall_d = -np.dot(wall_normal_vec, wall_point)

                vertices = get_aabb_vertices(aabb)
                dists_vert2wall = [
                    distance_to_plane(vert, [*wall_normal_vec, wall_d], keep_sign=True)
                    for vert in vertices
                ]
                dist_max = max(dists_vert2wall)
                dist_min = min(dists_vert2wall)
                projected_points_2d = project_vertices_to_plane(
                    np.array(vertices),
                    [*wall_normal_vec, wall_d],
                )
                projected_polygon = create_polygon_from_vertices(projected_points_2d)
                wall_objs_info.append(
                    {
                        "name": name,
                        "vertices": vertices,
                        "wall_dist_min": dist_min,
                        "wall_dist_max": dist_max,
                        "polygon": projected_polygon,
                    }
                )

            possible_objs_on_wall = get_possible_obj_on_wall(
                wall_objs_info,
                is_lateral_wall=is_lateral_wall,
                verbose=False,
                visualize=False,
            )
            for possible_obj_name in possible_objs_on_wall:
                objs_possible_mounts[possible_obj_name].append(wall_mask_path)

        if self.verbose:
            log.info("\nInferring mounting types...\n")

        wall_mount_count = {}
        mount_info = []
        n_objects = len(self.names)

        for i, (mask, name, clean_caption) in enumerate(
            zip(self.masks, self.names, self.phrases_recaptioned)
        ):
            obj_mount_info = {"floor": True, "wall": None}

            if self.infer_mounting_type:
                if self.verbose:
                    log.info("-----------------")
                    log.info(f"Object {i + 1} / {n_objects}")
                    log.info(f"Name: {name}")
                    cand_wall_fnames = [
                        wall_fpath.split("/")[-1].split(".")[0]
                        for wall_fpath in objs_possible_mounts[name]
                    ]
                    log.info(f"Candidate walls: {cand_wall_fnames}")

                nonprojected_img_path = (
                    f"{self.segmentation_dir}/{name}_nonprojected.png"
                )
                annotated_bbox_img_path = (
                    f"{self.segmentation_dir}/{name}_annotated_bboxes.png"
                )
                nonprojected_obj_with_all_wall_img_path = (
                    f"{self.segmentation_dir}/{name}_walls_nonprojected.png"
                )

                all_cand_walls_mask = np.full(
                    (self.new_height, self.new_width),
                    False,
                    dtype=bool,
                )
                for wall_fpath in objs_possible_mounts[name]:
                    cand_wall_mask = np.array(Image.open(wall_fpath))
                    all_cand_walls_mask = all_cand_walls_mask | cand_wall_mask
                merged_obj_with_all_wall_mask = all_cand_walls_mask | mask
                nonprojected_obj_with_all_wall_img = self.rgb * np.expand_dims(
                    merged_obj_with_all_wall_mask,
                    axis=-1,
                )
                Image.fromarray(nonprojected_obj_with_all_wall_img).save(
                    nonprojected_obj_with_all_wall_img_path
                )

                obj_cand_color_walls = [
                    raw_mask_path.replace("_mask.png", "_nonprojected.png")
                    for raw_mask_path in objs_possible_mounts[name]
                ]

                if obj_cand_color_walls:
                    if self.verbose:
                        log.info("Inferring mounting type...")

                    mount_select_payload = self.gpt.payload_mount_type(
                        caption=clean_caption,
                        bbox_img_path=annotated_bbox_img_path,
                        obj_and_wall_mask_path=nonprojected_obj_with_all_wall_img_path,
                        candidates_fpaths=obj_cand_color_walls,
                    )
                    resp = self.gpt(payload=mount_select_payload, verbose=self.verbose)
                    if resp is None:
                        self.mount_info = None
                        self.wall_mount_count = None
                        return
                    resp = resp.lower()

                    if "wall" in resp:
                        is_on_floor = "floor" in resp
                        mount_walls = []
                        mount_bases = resp.split(",")
                        for base_obj in mount_bases:
                            if "wall" in base_obj:
                                wall_idx = int(base_obj[len("wall"):].strip())
                                selected_wall_path = obj_cand_color_walls[
                                    wall_idx
                                ].replace("_nonprojected.png", "_mask.png")
                                mount_walls.append(selected_wall_path)
                                wall_mount_count[selected_wall_path] = (
                                    wall_mount_count.get(selected_wall_path, 0) + 1
                                )
                        obj_mount_info["floor"] = is_on_floor
                        obj_mount_info["wall"] = mount_walls

                    elif self.infer_aligned_wall:
                        if self.verbose:
                            log.info("Inferring aligned wall...")

                        align_wall_select_payload = self.gpt.payload_align_wall(
                            caption=clean_caption,
                            bbox_img_path=annotated_bbox_img_path,
                            nonproject_obj_img_path=nonprojected_img_path,
                            obj_and_wall_mask_path=nonprojected_obj_with_all_wall_img_path,
                            candidates_fpaths=obj_cand_color_walls,
                        )
                        resp = self.gpt(payload=align_wall_select_payload, verbose=self.verbose)
                        if resp is None:
                            self.mount_info = None
                            self.wall_mount_count = None
                            return
                        resp = resp.lower()

                        if "wall" in resp:
                            mount_walls = []
                            mount_bases = resp.split(",")
                            for base_obj in mount_bases:
                                if "wall" in base_obj:
                                    wall_idx = int(base_obj[len("wall"):].strip())
                                    selected_wall_path = obj_cand_color_walls[
                                        wall_idx
                                    ].replace("_nonprojected.png", "_mask.png")
                                    mount_walls.append(selected_wall_path)
                                    wall_mount_count[selected_wall_path] = (
                                        wall_mount_count.get(selected_wall_path, 0)
                                        + 1
                                    )
                            obj_mount_info["floor"] = True
                            obj_mount_info["wall"] = mount_walls

            mount_info.append(obj_mount_info)

        for obj_mount_info in mount_info:
            if obj_mount_info["wall"] and len(obj_mount_info["wall"]) > 1:
                obj_wall_count = [
                    (wall_i, wall_mount_count.get(wall_name, 0))
                    for wall_i, wall_name in enumerate(obj_mount_info["wall"])
                ]
                obj_wall_count.sort(reverse=True, key=lambda x: x[1])
                frequent_wall_i = obj_wall_count[0][0]
                obj_mount_info["wall"].insert(
                    0,
                    obj_mount_info["wall"].pop(frequent_wall_i),
                )

        self.mount_info = mount_info
        self.wall_mount_count = wall_mount_count

    def _update_articulation_counts(self):
        articulation_counts = {}
        for i, caption in enumerate(self.phrases_recaptioned):
            if caption in self.detected_objs:
                articulated_info = self.detected_objs[caption]
            elif f"{caption}s" in self.detected_objs:
                articulated_info = self.detected_objs[f"{caption}s"]
            else:
                raise ValueError(
                    "Got invalid caption! Valid options are: "
                    f"{self.detected_objs.keys()}, got: {caption}(s)"
                )

            if articulated_info is not None:
                log.info("-------------------")
                log.info(f"articulated caption: {caption}")
                link_count_payload = self.gpt.payload_count_drawer_door(
                    caption=caption,
                    bbox_img_path=(
                        f"{self.segmentation_dir}/{self.names[i]}_annotated_bboxes.png"
                    ),
                    nonproject_obj_img_path=(
                        f"{self.segmentation_dir}/{self.names[i]}_nonprojected.png"
                    ),
                )
                resp = self.gpt(payload=link_count_payload, verbose=self.verbose)
                if resp is None:
                    self.articulation_counts = None
                    return

                n_doors = int(resp.split(" door")[0][-1]) if "door" in resp else 0
                n_drawers = int(resp.split(" drawer")[0][-1]) if "drawer" in resp else 0

                articulation_counts[i] = (n_doors, n_drawers)
                log.info(f"new (door, drawer) count: {(n_doors, n_drawers)}")
                log.info("-------------------\n")

        self.articulation_counts = articulation_counts

    def _save_outputs(self):
        self.detected_categories_path = (
            f"{self.save_dir}/step_1_detected_categories.json"
        )
        step_1_output_info = {
            "K": self.camera_intrinsics_matrix,
            "detected_categories": self.detected_categories_path,
            "floor_mask": self.floor_mask_path,
            "wall_mask_planes": self.all_wall_mask_planes,
            "z_direction": self.z_dir,
            "origin_pos": self.origin_pos,
            "input_rgb": self.input_path,
            "input_depth": self.depth_path,
            "depth_limits": self.depth_limits,
        }
        self.step_1_output_path = f"{self.save_dir}/step_1_output_info.json"
        with open(self.step_1_output_path, "w+") as f:
            json.dump(step_1_output_info, f, indent=4, cls=NumpyTorchEncoder)

        info = {
            "names": self.names,
            "phrases": self.phrases,
            "phrases_recaptioned": self.phrases_recaptioned,
            "segmentation_dir": self.segmentation_dir,
            "articulation_counts": self.articulation_counts,
            "boxes": self.boxes.cpu().numpy(),
            "logits": self.logits.cpu().numpy(),
            "mount": self.mount_info,
        }
        with open(self.detected_categories_path, "w+") as f:
            json.dump(info, f, indent=4, cls=NumpyTorchEncoder)
