# Standard Libraries
import os
import json
from pathlib import Path
from copy import deepcopy

# Third-party Libraries
import torch as th
import numpy as np
from PIL import Image
import imageio
import cv2
from loguru import logger as log

# OmniGibson Libraries
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn

# Local / Project Utilities
from our_method.utils.processing_utils import prepare_output_dir, NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from our_method.utils.scene_utils import create_scene, take_photo, compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory
from our_method.utils.physics_settle import resolve_cfg as resolve_physics_settle_cfg, settle_scene
import our_method.utils.transform_utils as T


class RealSceneGenerator:
    """
    3rd Step in ACDC pipeline. This takes in the output from Step 2 (Digital Cousin Matching) and generates
    fully populated digital cousin scenes

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 2, which includes the following:
            - Per-object (category,, model, pose) digital cousin information

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """
    
    # Set of non-collidable categories
    NON_COLLIDABLE_CATEGORIES = {
        "towel",
        "rug",
        "mirror",
        "picture",
        "painting",
        "window",
        "art",
    }

    CATEGORIES_MUST_ON_FLOOR = {
        "rug",
        "carpet"
    }
    
    SAMPLING_METHODS = {
        "random",
        "ordered",
    }

    # support_yaw_align 기본값. config 에서 준 키만 덮어쓴다.
    SUPPORT_YAW_ALIGN_DEFAULTS = {
        "enabled": True,
        "min_rect_fill": 0.9,        # 위에서 본 외곽선(convex hull)이 최소 외접 사각형을 채우는 비율. 원형은 ~0.785
        "snap_tol_deg": 10.0,        # 외곽선 yaw 가 지지 물체 yaw(90도 단위)와 이 안이면 지지 물체에 붙인다
        "support_gap_m": [-0.05, 0.10],  # (물체 바닥 - 지지 물체 윗면) 허용 범위
        "min_aspect": 1.15,          # 장단변 비가 이보다 크면 장축 방향으로 90도 모호성을 푼다
        "categories": None,          # None 이면 규칙에 맞는 모든 물체. 리스트면 그 카테고리만
        "flip": "match",             # 180도 모호성 해소: "match"(뷰 매칭 yaw 에 가까운 쪽) | "gpt"(두 뷰를 GPT 에 물음)
        "gpt_version": "5.1",
        # 정렬이 끝난 yaw 에 카테고리별로 더하는 각도(도). {"electric_stove": 180} 처럼 준다.
        # 입력 사진에 앞뒤 단서가 없을 때(버튼이 안 보이는 검은 판 등) 알고 있는 방향을 넣는 용도다.
        "post_offset_deg": None,
        # 받침 물체가 없는(바닥에 선) 물체도 정렬할 카테고리. 바닥과 평행 맞춤은 건너뛰고, 외곽선 장축으로
        # 90도 모호성만 푼다. 거의 정사각형인 책상(0.65 x 0.55)은 뷰 매칭이 90도 틀리기 쉽다.
        "floor_categories": None,
        "floor_min_rect_fill": 0.8,  # 바닥 물체는 앞 물체에 가려 외곽선이 덜 찬다
        "floor_min_aspect": 1.05,    # 바닥 물체의 90도 모호성 판정 장단변 비. 받침대 0.65x0.55(1.18)가 가려서 1.14 로 잡힌다
    }

    def __init__(
            self,
            verbose=False,
    ):
        """
        Args:
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.verbose = verbose
        # Instance variables to be set in __call__ or helper methods
        self.n_scenes = 0
        self.sampling_method = "ordered"
        self.resolve_collision = True
        self.discard_objs = None
        self.save_dir = None
        self.visualize_scene = False
        self.visualize_scene_tilt_angle = 0
        self.visualize_scene_radius = 5
        self.save_visualization = True

        self.step_1_output_path = None
        self.step_2_output_path = None
        self.step_2_output_info = None
        
        # Loaded data from Step 1 & 2
        self.n_cousins = 0
        self.n_objects = 0
        self.cousins = {}
        self.rgb = None
        self.h = 0
        self.w = 0
        self.K = None
        self.z_dir = None
        self.wall_mask_planes = None
        self.cam_pos = None
        self.cam_quat = None
        self.pc = None
        self.detected_categories = None


    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            camera_info=None,
            n_scenes=1,
            sampling_method="ordered",
            resolve_collision=True,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=1,
            save_visualization=True,
            save_camera_info_extrinsic=False,
            snap_yaw_deg=None,
            snap_yaw_categories=None,
            yaw_offset_deg=None,
            support_yaw_align=None,
            physics_settle=None,
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:
        ...
        """
        # Store all input parameters as instance variables
        self.step_1_output_path = step_1_output_path
        self.step_2_output_path = step_2_output_path
        self.camera_info = camera_info
        self.n_scenes = n_scenes
        self.sampling_method = sampling_method
        self.resolve_collision = resolve_collision
        self.visualize_scene = visualize_scene
        self.visualize_scene_tilt_angle = visualize_scene_tilt_angle
        self.visualize_scene_radius = visualize_scene_radius
        self.save_visualization = save_visualization
        self.save_camera_info_extrinsic = save_camera_info_extrinsic
        # 원본 씬이 축 정렬이면 최종 yaw 를 격자에 붙여 매칭 노이즈를 없앨 수 있다.
        # None/0 이면 끈다 (기존 동작과 동일).
        # snap_yaw_categories 를 주면 그 카테고리에만 적용한다 (None 이면 전체).
        self.snap_yaw_deg = snap_yaw_deg
        self.snap_yaw_categories = set(snap_yaw_categories) if snap_yaw_categories else None
        # 카테고리별 yaw 강제 오프셋 (도 단위). {"stove": 90} 처럼 준다.
        # 뷰 매칭이 특정 물체에서만 크게 어긋날 때 그 물체만 돌려놓는 용도다.
        self.yaw_offset_deg = dict(yaw_offset_deg) if yaw_offset_deg else {}
        # 지지 물체 위에 놓인 직사각형 물체의 yaw 를 뷰 매칭 대신 위에서 본 외곽선과 지지 물체로 정한다.
        # 인덕션처럼 위에서 보면 대칭인 판은 DINOv2 매칭이 yaw 를 못 가리고, 위에 놓인 물체가 마스크
        # 가운데를 가려도 외곽선(convex hull)은 그대로 남는다. None/false 면 끈다 (기존 동작과 동일).
        if support_yaw_align is True:
            support_yaw_align = {}
        if isinstance(support_yaw_align, dict) and support_yaw_align.get("enabled", True):
            self.support_yaw_align = {**self.SUPPORT_YAW_ALIGN_DEFAULTS, **support_yaw_align}
        else:
            self.support_yaw_align = None
        self.footprints = {}
        # 저장 직전 관통 해소 + 중력 안착 (our_method/utils/physics_settle.py). None/false 면 끈다 (기존 동작과 동일).
        self.physics_settle = resolve_physics_settle_cfg(physics_settle)

        # Load step 2 info
        with open(self.step_2_output_path, "r") as f:
            self.step_2_output_info = json.load(f)
        
        # Load relevant information from prior steps
        self.n_cousins = self.step_2_output_info["metadata"]["n_cousins"]
        self.cousins = self.step_2_output_info["objects"]
        self.n_objects = self.step_2_output_info["metadata"]["n_objects"]

        # Sanity check
        assert self.sampling_method in self.SAMPLING_METHODS, \
            f"Got invalid sampling_method! Valid methods: {self.SAMPLING_METHODS}, got: {self.sampling_method}"
        
        # Setup save directory and discard list
        self.save_dir = prepare_output_dir(self.step_2_output_path, save_dir, "step_3_output")
        if discard_objs:
            self.discard_objs = set(discard_objs.split(","))

        if self.verbose:
            log.info(f"Generating simulated scenes given output {self.step_2_output_path}...")
            log.debug("Generating simulated scenes in OmniGibson")

        # Load input data and compute 3D context
        self._load_and_setup_environment()
        if self.support_yaw_align is not None:
            self._compute_footprints()

        # Launch omnigibson
        og.launch()

        # Loop over all sample indices to generate individual scenes
        for scene_count in range(self.n_scenes):
            log.info(f"[Scene {scene_count + 1} / {self.n_scenes}]")

            scene_save_dir = f"{self.save_dir}/scene_{scene_count}"
            Path(scene_save_dir).mkdir(parents=True, exist_ok=True)
   
            # Determine which cousin model to use
            cousin_idxs = self._select_cousin_indices(scene_count)
            
            # --- 1. Initial Object Processing and Alignment ---
            scene_info = self._process_individual_objects(
                scene_count=scene_count,
                scene_save_dir=scene_save_dir,
                cousin_idxs=cousin_idxs,
            )

            # --- 2. Scene Refinement, Collision, and Final Placement ---
            self._refine_scene_and_resolve_physics(
                scene_count=scene_count,
                scene_save_dir=scene_save_dir,
                scene_info=scene_info,
            )

        # Compile final results across all scenes and save to disk
        step_3_output_path = self._save_step3_outputs()

        log.success("Completed Simulated Scene Generation!")

        return True, step_3_output_path

    # --- Private Helper Methods ---

    def _load_and_setup_environment(self):
        """Loads input data, computes point cloud, and camera pose."""
        with open(self.step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            self.detected_categories = json.load(f)

        seg_dir = self.detected_categories["segmentation_dir"]
        self.K = np.array(step_1_output_info["K"])
        self.step_1_input_rgb = step_1_output_info["input_rgb"]
        self.rgb = np.array(Image.open(step_1_output_info["input_rgb"]))
        raw_depth = np.array(Image.open(step_1_output_info["input_depth"]))
        depth_limits = np.array(step_1_output_info["depth_limits"])
        depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
        self.pc = compute_point_cloud_from_depth(depth=depth, K=self.K)
        self.h, self.w, _ = self.rgb.shape

        self.z_dir = np.array(step_1_output_info["z_direction"])
        self.wall_mask_planes = step_1_output_info["wall_mask_planes"]
        origin_pos = np.array(step_1_output_info["origin_pos"])
        if self.save_camera_info_extrinsic:
            self.w = self.camera_info['intrinsics']['image_width']
            self.h = self.camera_info['intrinsics']['image_height']
            self.cam_pos = self.camera_info['camera']['position']
            self.cam_quat = self.camera_info['camera']['orientation']
        else:
            self.cam_pos, self.cam_quat = compute_relative_cam_pose_from(z_dir=self.z_dir, origin_pos=origin_pos)

    def _select_cousin_indices(self, scene_count):
        """Determines the cousin index for each object based on the sampling method."""
        if self.sampling_method == "random":
            cousin_idxs = dict()
            for obj_name in self.cousins.keys():
                cousin_idxs[obj_name] = np.random.randint(0, self.n_cousins)
        elif self.sampling_method == "ordered":
            cousin_idxs = {obj_name: scene_count for obj_name in self.cousins.keys()}
        else:
            raise ValueError(f"sampling_method {self.sampling_method} not supported!")
        return cousin_idxs

    def _process_individual_objects(
        self,
        scene_count, scene_save_dir, cousin_idxs,
    ):
        """Loads and aligns each object individually, saving per-object scene info."""
        scene = create_scene(floor=False)
        og.sim.viewer_camera.image_width = self.w
        og.sim.viewer_camera.image_height = self.h
        og.sim.viewer_camera.set_position_orientation(th.tensor(self.cam_pos, dtype=th.float), th.tensor(self.cam_quat, dtype=th.float))
        
        seg_dir = self.detected_categories["segmentation_dir"]

        for obj_idx, (obj_name, obj_cousin_idx) in enumerate(cousin_idxs.items()):
            if self.verbose:
                log.info(f"[Scene {scene_count + 1} / {self.n_scenes}] [Object {obj_idx + 1} / {self.n_objects}] generating...")
            if self.discard_objs and obj_name in self.discard_objs:
                continue

            obj_info = self.step_2_output_info["objects"][obj_name]
            is_articulated = obj_info["articulated"]
            
            pc_obj = self._load_obj_pc(obj_name, verbose=self.verbose)
            cousin_info = obj_info["cousins"][obj_cousin_idx]

            # Import the cousin asset
            with og.sim.stopped():
                obj = DatasetObject(
                    name=obj_name, category=cousin_info["category"], model=cousin_info["model"], visual_only=True
                )
                scene.add_object(obj)
            og.sim.step()

            # Determine the reprojection offset
            pan_angle_offset, _ = get_reproject_offset(
                pc_obj=deepcopy(pc_obj), z_dir=self.z_dir, xy_dist=2.30, z_dist=0.65
            )

            take_photo(n_render_steps=50)
            # Align object model to point cloud
            # 뷰 매칭은 3.6도 간격이라 축 정렬 물체도 수십 도씩 어긋날 수 있다.
            # snap_yaw_deg 가 주어지면 재투영 보정까지 끝낸 최종 yaw 를 그 격자에 붙인다.
            final_z_angle = cousin_info["z_angle"] + pan_angle_offset

            extra = self.yaw_offset_deg.get(cousin_info["category"])
            if extra:
                if self.verbose:
                    log.info(f"  yaw offset [{cousin_info['category']}]: "
                             f"{np.rad2deg(final_z_angle):+.1f}° {extra:+.0f}° -> "
                             f"{np.rad2deg(final_z_angle) + extra:+.1f}°")
                final_z_angle += np.deg2rad(extra)

            snap_this = bool(self.snap_yaw_deg) and (
                self.snap_yaw_categories is None
                or cousin_info["category"] in self.snap_yaw_categories
            )
            if snap_this:
                step = np.deg2rad(self.snap_yaw_deg)
                snapped = round(final_z_angle / step) * step
                if self.verbose:
                    log.info(f"  yaw snap: {np.rad2deg(final_z_angle):+.1f}° -> "
                             f"{np.rad2deg(snapped):+.1f}° (격자 {self.snap_yaw_deg}°)")
                final_z_angle = snapped

            # 지지 물체 기준 yaw 정렬. 적용되면 align_model_pose 의 점군 기반 yaw 보정은 끈다.
            support_yaw_info = None
            if self.support_yaw_align is not None:
                support_yaw_info = self._support_aligned_yaw(obj_name, obj, cousin_info, final_z_angle,
                                                             pan_angle_offset)
                if support_yaw_info is not None:
                    final_z_angle = support_yaw_info["z_angle"]

            obj_scale, obj_bbox_extent, tf_from_cam = align_model_pose(
                obj=obj, pc_obj=pc_obj, obj_z_angle=final_z_angle,
                obj_ori_offset=cousin_info["ori_offset"], z_dir=deepcopy(self.z_dir),
                cam_pos=self.cam_pos, cam_quat=self.cam_quat, is_articulated=is_articulated, verbose=self.verbose,
                refine_yaw=support_yaw_info is None,
            )
            
            take_photo(n_render_steps=50)
            wall_mount_fpaths = self.detected_categories["mount"][obj_idx]["wall"]
            if wall_mount_fpaths is not None:
                # Align object with the wall
                for mount_wall_idx, wall_mount_fpath in enumerate(wall_mount_fpaths):
                    obj_scale, obj_bbox_extent, tf_from_cam = align_obj_with_wall(
                        obj=obj, cam_pos=self.cam_pos, cam_quat=self.cam_quat,
                        wall_normal=self.wall_mask_planes[wall_mount_fpath]["normal"],
                        wall_point=self.wall_mask_planes[wall_mount_fpath]["point"],
                        wall_is_vertical=True, resize_only=mount_wall_idx > 0,
                    )
            take_photo(n_render_steps=100)

            # Save per-object scene info
            obj_save_dir = f"{scene_save_dir}/{obj_name}"
            Path(obj_save_dir).mkdir(parents=True, exist_ok=True)
            obj_scene_info = {
                "category": obj.category, "model": obj.model, "scale": obj_scale,
                "bbox_extent": obj_bbox_extent, "tf_from_cam": tf_from_cam,
                "mount": self.detected_categories["mount"][obj_idx],
            }
            if support_yaw_info is not None:
                obj_scene_info["support_yaw_align"] = support_yaw_info
            with open(f"{obj_save_dir}/{obj_name}_scene_info.json", "w+") as f:
                json.dump(obj_scene_info, f, indent=4, cls=NumpyTorchEncoder)

            take_photo()
            scene.remove_object(obj)

        # Compile and return scene info dictionary
        scene_info = {"resolution": [self.h, self.w], "cam_pose": [self.cam_pos, self.cam_quat], "objects": {}}
        for obj_name in cousin_idxs.keys():
            if self.discard_objs and obj_name in self.discard_objs:
                continue
            with open(f"{scene_save_dir}/{obj_name}/{obj_name}_scene_info.json", "r") as f:
                scene_obj_info = json.load(f)
            scene_info["objects"][obj_name] = scene_obj_info
        
        scene_info["scene_graph"] = f"{scene_save_dir}/scene_{scene_count}_graph.json"

        return scene_info

    def _load_obj_pc(self, obj_name, verbose=False):
        """Returns the (N, 3) camera-frame point cloud of @obj_name from its pruned mask."""
        seg_dir = self.detected_categories["segmentation_dir"]
        obj_mask = np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected_mask_pruned.png"))
        pc_obj = self.pc.reshape(-1, 3)[np.array(obj_mask).flatten().nonzero()[0]]

        # GAIA_PC_OUTLIER_FILTER=1 이면 마스크 경계가 배경을 물어 생긴 depth 이상치를 잘라낸다.
        # align_model_pose 는 점군의 min/max 로 AABB 를 잡으므로(scene_utils.py:170)
        # 배경에 걸린 점 몇 %만 있어도 크기와 위치가 함께 망가진다. 물체는 연속된
        # 하나의 덩어리이므로 median depth 에서 크게 떨어진 점은 배경으로 본다.
        # 허용 폭은 3*IQR 로 물체 두께에 맞춰 늘어나되 최소 0.15m 는 보장한다.
        # 변수를 주지 않으면 기존 동작 그대로다.
        if os.environ.get("GAIA_PC_OUTLIER_FILTER") and len(pc_obj) > 20:
            z = pc_obj[:, 2]
            q1, q3 = np.percentile(z, [25, 75])
            band = max(3.0 * (q3 - q1), 0.15)
            keep = np.abs(z - np.median(z)) <= band
            if keep.sum() >= 20 and keep.sum() < len(z):
                if verbose:
                    log.info(f"[{obj_name}] depth 이상치 {len(z) - keep.sum()}/{len(z)} 점 제거 "
                             f"(median {np.median(z):.3f}m, 허용 ±{band:.3f}m)")
                pc_obj = pc_obj[keep]
        return pc_obj

    @staticmethod
    def _wrap_angle(a, period=2 * np.pi):
        """Wraps @a into [-period / 2, period / 2)."""
        return (a + period / 2) % period - period / 2

    def _compute_footprints(self):
        """
        물체마다 중력 정렬(tilt 보정) 평면에서 본 외곽선과 높이 범위를 구한다.
        align_model_pose 와 같은 tilt 회전을 쓰므로, 여기서 구한 각도는 obj_z_angle 과 같은 좌표다
        (물체 로컬 x 축이 tilt 프레임 xy 평면에서 obj_z_angle 방향을 향한다).
        convex hull 을 쓰므로 위에 놓인 물체가 마스크 가운데를 가려도 외곽선은 변하지 않는다.
        """
        tilt_angle = np.arctan2(self.z_dir[1], self.z_dir[2])
        tilt_mat = T.euler2mat([tilt_angle, 0, 0])
        for obj_name in self.cousins.keys():
            pc_obj = self._load_obj_pc(obj_name)
            if len(pc_obj) < 20:
                continue
            pc_rot = pc_obj @ tilt_mat.T
            hull = cv2.convexHull(pc_rot[:, :2].astype(np.float32))
            box = cv2.boxPoints(cv2.minAreaRect(hull))
            edge_a, edge_b = box[1] - box[0], box[2] - box[1]
            len_a, len_b = float(np.linalg.norm(edge_a)), float(np.linalg.norm(edge_b))
            self.footprints[obj_name] = {
                "hull": hull,
                "center": box.mean(axis=0),
                "theta": float(np.arctan2(edge_a[1], edge_a[0])),   # edge_a 방향
                "len_a": len_a,
                "len_b": len_b,
                "area": len_a * len_b,
                "fill": float(cv2.contourArea(hull) / max(len_a * len_b, 1e-9)),
                "z_lo": float(np.percentile(pc_rot[:, 2], 5)),
                "z_hi": float(np.percentile(pc_rot[:, 2], 95)),
            }

    def _find_support(self, obj_name):
        """외곽선이 직사각형이고 @obj_name 바로 아래에서 그 중심을 받치는 물체 이름을 찾는다. 없으면 None."""
        cfg = self.support_yaw_align
        fp = self.footprints[obj_name]
        gap_lo, gap_hi = cfg["support_gap_m"]
        best, best_z = None, -np.inf
        for other, ofp in self.footprints.items():
            if other == obj_name or ofp["fill"] < cfg["min_rect_fill"] or ofp["area"] < 1.5 * fp["area"]:
                continue
            center = tuple(float(c) for c in fp["center"])
            if cv2.pointPolygonTest(ofp["hull"], center, False) < 0:
                continue
            gap = fp["z_lo"] - ofp["z_hi"]
            if gap_lo <= gap <= gap_hi and ofp["z_hi"] > best_z:
                best, best_z = other, ofp["z_hi"]
        return best

    def _support_aligned_yaw(self, obj_name, obj, cousin_info, matched_z_angle, pan_angle_offset):
        """
        지지 물체 위의 직사각형 물체에 대해 외곽선 기반 obj_z_angle 을 돌려준다. 해당하지 않으면 None.

        1. 외곽선의 최소 외접 사각형으로 yaw 를 구한다 (90도 모호).
        2. 지지 물체의 외곽선 yaw 와 snap_tol_deg 이내면 지지 물체에 평행하게 붙인다.
        3. 에셋 footprint 의 장축이 외곽선 장축과 맞는 쪽으로 90도 모호성을 푼다.
        4. 남은 180도 모호성은 flip 설정에 따라 뷰 매칭 yaw(@matched_z_angle)에 가까운 쪽을 고르거나,
           두 후보에 해당하는 에셋 뷰를 GPT 에 보여주고 고르게 한다.
        """
        cfg = self.support_yaw_align
        fp = self.footprints.get(obj_name)
        if fp is None or cousin_info["ori_offset"] is not None:
            return None
        if cfg["categories"] is not None and cousin_info["category"] not in cfg["categories"]:
            return None
        support = self._find_support(obj_name) if fp["fill"] >= cfg["min_rect_fill"] else None
        if support is None:
            if cousin_info["category"] not in (cfg["floor_categories"] or []) or fp["fill"] < cfg["floor_min_rect_fill"]:
                return None
            support = "floor"

        theta = fp["theta"]
        sup_theta, snapped = None, False
        if support != "floor":
            sup_theta = self.footprints[support]["theta"]
            delta = self._wrap_angle(sup_theta - theta, np.pi / 2)
            snapped = abs(np.rad2deg(delta)) <= cfg["snap_tol_deg"]
            if snapped:
                theta += delta

        # 에셋 footprint (로컬 x, y). 기본 자세에서 aabb 를 잰다.
        obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
        obj.keep_still()
        og.sim.step_physics()
        ex, ey = [float(v) for v in obj.aabb_extent[:2]]

        # k 짝수면 로컬 x 가 edge_a, 홀수면 edge_b 에 놓인다
        cands = [theta + k * np.pi / 2 for k in range(4)]
        errs = [abs(ex - fp["len_a"]) + abs(ey - fp["len_b"]) if k % 2 == 0
                else abs(ex - fp["len_b"]) + abs(ey - fp["len_a"]) for k in range(4)]
        asset_aspect = max(ex, ey) / max(min(ex, ey), 1e-6)
        fp_aspect = max(fp["len_a"], fp["len_b"]) / max(min(fp["len_a"], fp["len_b"]), 1e-6)
        min_aspect = cfg["floor_min_aspect"] if support == "floor" else cfg["min_aspect"]
        if asset_aspect >= min_aspect and fp_aspect >= min_aspect:
            best_parity = int(np.argmin(errs)) % 2
            cands = [c for k, c in enumerate(cands) if k % 2 == best_parity]
        cands = sorted(cands, key=lambda c: abs(self._wrap_angle(c - matched_z_angle)))
        z_angle = cands[0]
        flip_source = "match"
        if cfg["flip"] == "gpt":
            # 남은 후보 중 매칭에 가장 가까운 것과 그 180도 반대편을 비교한다
            pair = [cands[0], cands[0] + np.pi]
            choice = self._ask_gpt_flip(obj_name, cousin_info, pair, pan_angle_offset)
            if choice is not None:
                z_angle = pair[choice]
                flip_source = "gpt"
        post_offset = (cfg["post_offset_deg"] or {}).get(cousin_info["category"])
        if post_offset:
            z_angle += np.deg2rad(post_offset)
            flip_source += f"{post_offset:+.0f}deg"
        z_angle = float(self._wrap_angle(z_angle))

        info = {
            "support": support,
            "footprint_deg": float(np.rad2deg(fp["theta"])),
            "support_footprint_deg": None if sup_theta is None else float(np.rad2deg(sup_theta)),
            "snapped_to_support": bool(snapped),
            "footprint_size": [fp["len_a"], fp["len_b"]],
            "footprint_fill": fp["fill"],
            "asset_xy_extent": [ex, ey],
            "matched_z_angle_deg": float(np.rad2deg(matched_z_angle)),
            "flip_source": flip_source,
            "z_angle_deg": float(np.rad2deg(z_angle)),
            "z_angle": z_angle,
        }
        if self.verbose:
            log.info(f"  support yaw align [{obj_name} on {support}]: 외곽선 {info['footprint_deg']:+.1f}° "
                     f"(지지 {info['support_footprint_deg'] if sup_theta is None else round(info['support_footprint_deg'], 1)}°, snap={snapped}), "
                     f"외곽선 {fp['len_a']:.3f}x{fp['len_b']:.3f} / 에셋 {ex:.3f}x{ey:.3f}, "
                     f"매칭 {info['matched_z_angle_deg']:+.1f}° -> {info['z_angle_deg']:+.1f}° (앞뒤: {flip_source})")
        return info

    def _ask_gpt_flip(self, obj_name, cousin_info, z_angles, pan_angle_offset):
        """
        @z_angles (180도 차이 나는 두 obj_z_angle) 에 해당하는 에셋 뷰 스냅샷을 GPT 에 보여주고 입력 사진과
        앞뒤가 맞는 쪽의 인덱스를 받는다. 실패하면 None.

        Step 2 와 같은 각도 규약을 쓴다: 뷰 i 의 z_angle = i * 2pi/100 - pi, 그리고
        Step 3 의 obj_z_angle = 뷰 z_angle + pan_angle_offset.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.warning(f"  [{obj_name}] OPENAI_API_KEY 가 없어 앞뒤 판단을 매칭 yaw 로 대신한다")
            return None
        from our_method.models.gpt import GPT

        view_dir = os.path.dirname(cousin_info["snapshot"])
        step = 2 * np.pi / 100
        cand_paths = []
        for z in z_angles:
            view_idx = int(round((self._wrap_angle(z - pan_angle_offset) + np.pi) / step)) % 100
            cand_paths.append(f"{view_dir}/{cousin_info['model']}_{view_idx}.png")

        seg_dir = self.detected_categories["segmentation_dir"]
        step_2_dir = os.path.dirname(self.step_2_output_path)
        gpt = GPT(api_key=api_key, version=self.support_yaw_align["gpt_version"], log_dir_tail="_GAIA")
        caption = obj_name.rsplit("_", 1)[0].replace("_", " ")
        payload = gpt.payload_nearest_neighbor_pose(
            caption=caption,
            img_path=self.step_1_input_rgb,
            bbox_img_path=f"{step_2_dir}/{obj_name}/top_k_model_candidates/{obj_name}_annotated_bboxes.png",
            nonproject_obj_img_path=f"{seg_dir}/{obj_name}_nonprojected.png",
            candidates_fpaths=cand_paths,
        )
        # 두 후보는 같은 외곽선의 180도 반대 자세다. 위에 놓인 물체가 단서가 될 수 있다고 알려준다.
        on_top = [n for n in self.footprints if n != obj_name and self._find_support(n) == obj_name]
        note = ("Note: both candidates already have the correct footprint alignment; they differ only by a "
                "180-degree flip (front/back swapped). Decide which end of the asset faces the camera. ")
        if on_top:
            note += (f"In the input image, {', '.join(n.rsplit('_', 1)[0].replace('_', ' ') for n in on_top)} "
                     "rests on top of the target and occludes part of it. Where it rests is a strong cue: such "
                     "objects usually sit on the functional area of the target (e.g., cookware on the heating zone), "
                     "and the uncovered part shows where the other features are.")
        payload["input"][0]["content"].insert(-1, {"type": "input_text", "text": note})

        resp = gpt(payload=payload, verbose=self.verbose)
        if resp is None:
            return None
        digits = [int(c) for c in resp if c.isdigit()]
        if not digits or digits[0] not in (0, 1):
            log.warning(f"  [{obj_name}] GPT 앞뒤 응답을 해석하지 못함: {resp!r}")
            return None
        if self.verbose:
            log.info(f"  [{obj_name}] GPT 앞뒤 선택: {digits[0]} ({os.path.basename(cand_paths[digits[0]])}, "
                     f"후보 {[os.path.basename(p) for p in cand_paths]})")
        return digits[0]

    def _refine_scene_and_resolve_physics(
        self,
        scene_count, scene_save_dir, scene_info,
    ):
        """Loads full scene, computes scene graph, resolves collisions, and visualizes."""
        scene = RealSceneGenerator.load_cousin_scene(scene_info=scene_info, visual_only=True)
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {scene_info['resolution'][0]}] refining scene graph...")

        # --- 1. Infer Scene Graph and Adjust Height (Z-offset) ---
        scene_graph_info, all_obj_bbox_info, sorted_z_obj_bbox_info, final_scene_info = \
            self._infer_scene_graph_and_adjust_height(scene, scene_info)
        
        # Save scene graph
        scene_graph_info_path = f"{scene_save_dir}/scene_{scene_count}_graph.json"
        with open(scene_graph_info_path, "w+") as f:
            json.dump(scene_graph_info, f, indent=4, cls=NumpyTorchEncoder)

        # --- 2. Collision Resolution (X/Y plane) ---
        sorted_x_obj_bbox_info = dict(sorted(sorted_z_obj_bbox_info.items(), key=lambda x: x[1]['lower'][0], reverse=True))
        obj_names = list(sorted_x_obj_bbox_info.keys())
        
        if self.resolve_collision:
            self._resolve_horizontal_collisions(scene_count, scene, obj_names, scene_graph_info, final_scene_info)
        else:
            if self.verbose:
                log.info(f"[Scene {scene_count + 1} / {1}] skip depenetrating collisions.")

        # --- 3. Final Placement (Vertical Drop) ---
        if self.physics_settle is None:
            self._resolve_vertical_placement(scene_count, scene, obj_names, scene_graph_info, all_obj_bbox_info, final_scene_info)

            # Take final physics step, then save visualization + info
            og.sim.step_physics()
        else:
            # _resolve_vertical_placement 는 Touching 이 늦게 잡혀 물체를 수 cm 씩 받침 안으로 내린다.
            # 대신 AABB 로 관통을 풀고 씬 전체를 중력으로 안착시킨다. 안착된 새 씬은 렌더링에만 쓴다.
            if self.verbose:
                log.info(f"[Scene {scene_count + 1} / {1}] physics settle...")
            scene, settle_report = settle_scene(scene, final_scene_info, self.physics_settle)
            with open(f"{scene_save_dir}/physics_settle_report.json", "w+") as f:
                json.dump(settle_report, f, indent=4, cls=NumpyTorchEncoder)
        
        # --- 4. Save Final Visualization and Info ---
        self._save_final_outputs(scene_count, scene_save_dir, final_scene_info)
        
        if self.visualize_scene:
            self._visualize_scene_video(
                scene, scene_save_dir
            )
        
        # Return final info (though it's saved to disk)
        return final_scene_info

    def _infer_scene_graph_and_adjust_height(self, scene, scene_info):
        """Infers object relationships and adjusts object height for vertical plausibility."""
        all_obj_bbox_info = {}
        for obj_name, obj_info in scene_info["objects"].items():
            if self.discard_objs and obj_name in self.discard_objs:
                continue
            obj = scene.object_registry("name", obj_name)
            obj_bbox_info = compute_obj_bbox_info(obj=obj)
            obj_bbox_info["articulated"] = self.step_2_output_info["objects"][obj_name]["articulated"]
            obj_bbox_info["mount"] = obj_info["mount"]
            all_obj_bbox_info[obj_name] = obj_bbox_info
        sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: x[1]['lower'][2]))

        scene_graph_info = {"floor": {"objOnTop": [], "objBeneath": None, "mount": {"floor": True, "wall": False}}}
        final_scene_info = deepcopy(scene_info)

        for name in sorted_z_obj_bbox_info:
            obj_name_beneath, z_offset = compute_object_z_offset(
                target_obj_name=name, sorted_obj_bbox_info=sorted_z_obj_bbox_info, verbose=self.verbose,
            )
            obj = scene.object_registry("name", name)

            if scene_info["objects"][name]["category"] in self.CATEGORIES_MUST_ON_FLOOR:
                obj_name_beneath = "floor"
                z_offset = -sorted_z_obj_bbox_info[name]["lower"][-1]

            # Update Scene Graph
            scene_graph_info.setdefault(name, {"objOnTop": [], "objBeneath": obj_name_beneath, "mount": None})
            scene_graph_info.setdefault(obj_name_beneath, {"objOnTop": [], "objBeneath": None, "mount": None})
            scene_graph_info[name]["objBeneath"] = obj_name_beneath
            scene_graph_info[obj_name_beneath]["objOnTop"].append(name)
            scene_graph_info[name]["mount"] = scene_info["objects"][name]["mount"]
            obj.keep_still()

            # Apply Z-offset
            if z_offset != 0:
                mount_type = scene_info["objects"][name]["mount"]
                if not mount_type["floor"] and z_offset <= 0: continue
                new_center = sorted_z_obj_bbox_info[name]["center"] + np.array([0.0, 0.0, z_offset])
                obj.set_bbox_center_position_orientation(position=th.tensor(new_center, dtype=th.float), orientation=None)
                og.sim.step_physics()
                sorted_z_obj_bbox_info[name].update(compute_obj_bbox_info(obj=obj))
                if self.verbose and obj_name_beneath in sorted_z_obj_bbox_info:
                    gap = sorted_z_obj_bbox_info[name]["lower"][2] - sorted_z_obj_bbox_info[obj_name_beneath]["upper"][2]
                    log.info(f"  height adjust [{name} on {obj_name_beneath}]: z_offset {z_offset * 1000:+.1f}mm "
                             f"-> 간격 {gap * 1000:+.1f}mm")

            # Update relative transformation
            obj_pos, obj_quat = obj.get_position_orientation()
            rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
            final_scene_info["objects"][name]["tf_from_cam"] = T.pose2mat(rel_tf)

        for obj in scene.objects: obj.keep_still()
        og.sim.step_physics()
        return scene_graph_info, all_obj_bbox_info, sorted_z_obj_bbox_info, final_scene_info

    def _resolve_horizontal_collisions(self, scene_count, scene, obj_names, scene_graph_info, final_scene_info):
        """Resolves collisions between objects in the X/Y plane by moving them apart."""
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {1}] depenetrating collisions...")

        for obj1_idx, obj1_name in enumerate(obj_names):
            if any(cat in obj1_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue

            obj1 = scene.object_registry("name", obj1_name)
            obj1.keep_still()
            obj1.visual_only = False

            for obj2_name in obj_names[obj1_idx + 1:]:
                if any(cat in obj2_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue
                assert obj1_name != obj2_name

                if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                    continue
                
                obj2 = scene.object_registry("name", obj2_name)
                old_state = og.sim.dump_state()
                obj2.keep_still()
                obj2.visual_only = False
                og.sim.step_physics()

                if obj2.states[Touching].get_value(obj1):
                    if self.verbose:
                        log.info(f"Detected collision between {obj1_name} and {obj2_name}")
                    
                    obj2_ori_mat = T.quat2mat(obj2.get_position_orientation()[1].cpu().detach().numpy())
                    obj2_x_dir = obj2_ori_mat[:, 0]
                    obj2_y_dir = obj2_ori_mat[:, 1]
                    center_step_size = 0.01
                    obj2_to_obj1 = (obj1.get_position_orientation()[0] - obj2.get_position_orientation()[0]).cpu().detach().numpy()

                    chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                    center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis

                    while obj2.states[Touching].get_value(obj1):
                        og.sim.load_state(old_state)
                        new_center = obj2.get_position_orientation()[0] + th.tensor(center_step_dir, dtype=th.float) * center_step_size
                        obj2.set_position_orientation(position=new_center)
                        old_state = og.sim.dump_state()
                        og.sim.step_physics()

                    og.sim.load_state(old_state)
                    obj2.set_position_orientation(position=new_center)
                    obj_pos, obj_quat = obj2.get_position_orientation()
                    rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
                    final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                else:
                    og.sim.load_state(old_state)
                obj2.visual_only = True
            obj1.visual_only = True

    def _resolve_vertical_placement(self, scene_count, scene, obj_names, scene_graph_info, all_obj_bbox_info, final_scene_info):
        """Uses physics steps to gently place objects onto the surface beneath them."""
        if self.verbose:
            log.info(f"[Scene {scene_count + 1} / {1}] placing all objects down...")

        for obj in scene.objects: obj.keep_still()
        og.sim.step_physics()
        
        for obj1_name in obj_names:
            if any(cat in obj1_name for cat in self.NON_COLLIDABLE_CATEGORIES): continue
            
            if scene_graph_info[obj1_name]['objBeneath'] == "floor" or not all_obj_bbox_info[obj1_name]["mount"]["floor"]:
                continue
            
            obj_beneath_name = scene_graph_info[obj1_name]["objBeneath"]
            obj_beneath = scene.object_registry("name", obj_beneath_name)

            if "no_top" in obj_beneath.category or any(cat in obj_beneath_name for cat in self.NON_COLLIDABLE_CATEGORIES):
                continue
            
            obj1 = scene.object_registry("name", obj1_name)
            obj_beneath.keep_still()
            obj_beneath.visual_only = False
            old_state = og.sim.dump_state()

            obj1.keep_still()
            obj1.visual_only = False
            obj1_lower_corner, _ = obj1.aabb
            obj1_low_z = obj1_lower_corner[-1].item()
            obj_beneath_lower_corner, _ = obj_beneath.aabb
            obj_beneath_low_z = obj_beneath_lower_corner[-1].item()
            center_step_size = 0.005
            og.sim.step_physics()

            touching_at_start = obj1.states[Touching].get_value(obj_beneath)
            gap_at_start = obj1.aabb[0][-1].item() - obj_beneath.aabb[1][-1].item()
            if not touching_at_start:
                n_down = 0
                while obj1_low_z >= max(0, obj_beneath_low_z) and \
                    not obj1.states[Touching].get_value(obj_beneath):
                    og.sim.load_state(old_state)
                    new_center = obj1.get_position_orientation()[0] + th.tensor([0, 0, -1.0]) * center_step_size
                    obj1_low_z -= center_step_size
                    obj1.set_position_orientation(position=new_center)
                    old_state = og.sim.dump_state()
                    og.sim.step_physics()
                    n_down += 1

                og.sim.load_state(old_state)
                final_position = obj1.get_position_orientation()[0] - th.tensor([0, 0, -1.0]) * center_step_size
                obj1.set_position_orientation(position=final_position)
                obj_pos, obj_quat = obj1.get_position_orientation()
                rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), self.cam_pos, self.cam_quat)
                final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                if self.verbose:
                    log.info(f"  vertical place [{obj1_name} on {obj_beneath_name}]: 시작 간격 {gap_at_start * 1000:+.1f}mm, "
                             f"{n_down}번 내림 -> 간격 {(obj1.aabb[0][-1] - obj_beneath.aabb[1][-1]).item() * 1000:+.1f}mm")
            else:
                og.sim.load_state(old_state)
                if self.verbose:
                    log.info(f"  vertical place [{obj1_name} on {obj_beneath_name}]: 시작부터 접촉 "
                             f"(간격 {gap_at_start * 1000:+.1f}mm), 그대로 둠")

            obj_beneath.keep_still()
            obj1.keep_still()
            og.sim.step_physics()
            obj1.visual_only = True
            obj_beneath.visual_only = True

    def _save_final_outputs(self, scene_count, scene_save_dir, final_scene_info):
        """Saves the final RGB visualization and scene info JSON."""
        scene_rgb = take_photo(n_render_steps=10)
        H, W, _ = scene_rgb.shape
        resized_rgb = resize_image(self.rgb, height=H)
        concat_scene_rgb = np.concatenate([scene_rgb[:, :, :3], scene_rgb[:,:,:3]], axis=1)
        Image.fromarray(concat_scene_rgb).save(f"{scene_save_dir}/scene_{scene_count}_visualization.png")

        with open(f"{scene_save_dir}/scene_{scene_count}_info.json", "w+") as f:
            json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)

    def _save_step3_outputs(self) -> str:
        """
        Compiles and saves the final Step 3 output JSON across all generated scenes.
        """
        step_3_output_info = dict()
        for scene_count in range(self.n_scenes):
            scene_name = f"scene_{scene_count}"
            final_scene_info_path = f"{self.save_dir}/{scene_name}/{scene_name}_info.json"
            with open(final_scene_info_path, "r") as f:
                final_scene_info = json.load(f)
            step_3_output_info[scene_name] = final_scene_info

        step_3_output_path = f"{self.save_dir}/step_3_output_info.json"
        with open(step_3_output_path, "w+") as f:
            json.dump(step_3_output_info, f, indent=4, cls=NumpyTorchEncoder)

        return step_3_output_path


    def _visualize_scene_video(self, scene, scene_save_dir):
        """Generates and saves a rotating video visualization of the final scene."""
        og.sim.viewer_camera.add_modality('seg_semantic')
        aabb_points = []
        for obj in scene.objects:
            p1, p2 = obj.aabb
            aabb_points.append(p1)
            aabb_points.append(p2)
            if ToggledOn in obj.states:
                obj.states[ToggledOn].link.visible = False

        min_x = min([p[0] for p in aabb_points])
        min_y = min([p[1] for p in aabb_points])
        max_x = max([p[0] for p in aabb_points])
        max_y = max([p[1] for p in aabb_points])
        vis_cam_pos, vis_cam_ori = og.sim.viewer_camera.get_position_orientation()
        vis_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, vis_cam_pos[-1])
        cam_commands = get_vis_cam_trajectory(
            center_pos=vis_center, cam_pos=vis_cam_pos, cam_quat=vis_cam_ori,
            d_tilt=self.visualize_scene_tilt_angle, radius=self.visualize_scene_radius, n_steps=100
        )

        for _ in range(50):
            og.sim.render()

        if self.save_visualization:
            video_path = f"{scene_save_dir}/visualization_video.mp4"
            img_dir = f"{scene_save_dir}/scene_visualization"
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=20)
        for i, (pos, quat) in enumerate(cam_commands):
            og.sim.viewer_camera.set_position_orientation(pos, quat)
            og.sim.render()
            if self.save_visualization:
                obs, obs_info = og.sim.viewer_camera.get_obs()
                vis_rgb = obs["rgb"].cpu().detach().numpy()
                seg_semantic = obs["seg_semantic"].cpu().detach().numpy()

                filter_names = {"floors", "background"}
                filter_ids = {idn for idn, name in obs_info["seg_semantic"].items() if name in filter_names}
                seg_mask = np.ones_like(seg_semantic).astype(np.uint8) * 255
                for filter_id in filter_ids:
                    seg_mask[np.where(seg_semantic == filter_id)] = 0
                masked_vis_rgb = vis_rgb.astype(np.uint8)
                masked_vis_rgb[seg_mask == 0] = [0, 0, 0, 1]
                video_writer.append_data(masked_vis_rgb)
                
                vis_rgb[:, :, 3] = seg_mask
                Image.fromarray(vis_rgb).save(f"{img_dir}/vis_frame_{i}.png")
        if self.save_visualization:
            video_writer.close()

    # --- Static Helper Methods ---

    @staticmethod
    def load_cousin_scene(scene_info, visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath
        ...
        """
        # Stop sim, clear it, then load empty scene
        scene = create_scene(floor=True)

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(scene_info)

        # Set viewer camera to proper pose
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))

        # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                obj = DatasetObject(
                    name=obj_name,
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=visual_only,
                    scale=obj_info["scale"]
                )
                scene.add_object(obj)
                obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                    pose_A=np.array(obj_info["tf_from_cam"]),
                    pose_A_in_B=T.pose2mat(cam_pose),
                ))
                obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))
        
        # Initialize all objects by taking one step
        og.sim.step()
        return scene

    def joint_test(self, scene, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera
        ...
        """
        obj = scene.object_registry("name", "cabinet_0") 

        joint_index = 0
        positions = obj.get_joint_positions()

        for step in range(n_render_steps):
            # Change joint position every 10 steps (0 <-> 1.5)
            if step % 10 == 0:
                positions[joint_index] = 1.5 if positions[joint_index] == 0 else 0
                obj.set_joint_positions(positions)  # Update

            # Run physics simulation and render
            og.sim.step_physics()
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        return rgb