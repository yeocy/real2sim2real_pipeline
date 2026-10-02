import os
import json
import argparse
import numpy as np
import torch as th
from PIL import Image

import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
import omnigibson.utils.transform_utils as T


class VisualizeTwoScenes:
    def __init__(
        self,
        step3_output_path_a,
        step3_output_path_b,
        scene_idx_a=0,
        scene_idx_b=0,
        output=None,
        n_render_steps=100,
        visual_only=False,
        scene_b_offset=(2.0, 0.0, 0.0),   # 두 번째 scene을 옆으로 밀기
    ):
        self.step3_output_path_a = step3_output_path_a
        self.step3_output_path_b = step3_output_path_b
        self.scene_idx_a = scene_idx_a
        self.scene_idx_b = scene_idx_b
        self.output = output
        self.n_render_steps = n_render_steps
        self.visual_only = visual_only
        self.scene_b_offset = np.array(scene_b_offset, dtype=np.float32)

    def create_scene(self, floor=True, sky=True):
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    def _load_json_scene_info(self, json_path, scene_idx):
        with open(json_path, "r") as f:
            info = json.load(f)

        scene_key = f"scene_{scene_idx}"
        if scene_key not in info:
            raise ValueError(f"{scene_key} not found in {json_path}")
        return info[scene_key]

    def _add_objects_from_scene_info(
        self,
        scene,
        scene_info,
        name_prefix="a",
        world_offset=None,
    ):
        if world_offset is None:
            world_offset = th.zeros(3, dtype=th.float)
        else:
            world_offset = th.tensor(world_offset, dtype=th.float)

        cam_pose = scene_info["cam_pose"]

        cam_pos = th.tensor(cam_pose[0], dtype=th.float)
        cam_quat = th.tensor(cam_pose[1], dtype=th.float)
        cam_pose_mat = T.pose2mat((cam_pos, cam_quat))

        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                obj = DatasetObject(
                    name=f"{name_prefix}_{obj_name}",
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=self.visual_only,
                    scale=obj_info["scale"],
                )
                scene.add_object(obj)

                pose_A = th.tensor(obj_info["tf_from_cam"], dtype=th.float)

                obj_pose_world = T.pose_in_A_to_pose_in_B(
                    pose_A=pose_A,
                    pose_A_in_B=cam_pose_mat,
                )

                obj_pos, obj_quat = T.mat2pose(obj_pose_world)
                obj_pos = obj_pos + world_offset

                obj.set_position_orientation(obj_pos, obj_quat)

    def load_two_scenes(self):
        scene = self.create_scene(floor=True)

        scene_info_a = self._load_json_scene_info(self.step3_output_path_a, self.scene_idx_a)
        scene_info_b = self._load_json_scene_info(self.step3_output_path_b, self.scene_idx_b)

        # 카메라는 첫 번째 scene 기준으로 설정
        cam_pose_a = scene_info_a["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(
            th.tensor(cam_pose_a[0], dtype=th.float),
            th.tensor(cam_pose_a[1], dtype=th.float),
        )

        # 첫 번째 scene
        self._add_objects_from_scene_info(
            scene=scene,
            scene_info=scene_info_a,
            name_prefix="A",
            world_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        # 두 번째 scene (옆으로 이동)
        self._add_objects_from_scene_info(
            scene=scene,
            scene_info=scene_info_b,
            name_prefix="B",
            world_offset=self.scene_b_offset,
        )

        og.sim.step()
        return scene

    def take_photo(self, n_render_steps=5):
        for _ in range(n_render_steps):
            og.sim.render()

        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        return rgb

    def run(self):
        og.launch()

        self.load_two_scenes()
        scene_rgb = self.take_photo(n_render_steps=self.n_render_steps)

        output_path = self.output
        if output_path is None:
            out_dir = os.path.dirname(self.step3_output_path_a)
            output_path = os.path.join(out_dir, "two_scenes_visualization.png")

        Image.fromarray(scene_rgb).save(output_path)
        print(f"Two-scene visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--step3_output_path_a",
        type=str,
        required=True,
        help="Path to first step_3_output_info.json",
    )
    parser.add_argument(
        "--step3_output_path_b",
        type=str,
        required=True,
        help="Path to second step_3_output_info.json",
    )

    parser.add_argument("--scene_idx_a", type=int, default=0, help="Scene index for first json")
    parser.add_argument("--scene_idx_b", type=int, default=0, help="Scene index for second json")

    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--n_render_steps", type=int, default=1000000, help="Number of render steps")
    parser.add_argument("--visual_only", action="store_true", help="Load objects as visual only")

    parser.add_argument("--offset_x", type=float, default=0.0, help="X offset for second scene")
    parser.add_argument("--offset_y", type=float, default=0.0, help="Y offset for second scene")
    parser.add_argument("--offset_z", type=float, default=0.0, help="Z offset for second scene")

    args = parser.parse_args()

    vis = VisualizeTwoScenes(
        step3_output_path_a=args.step3_output_path_a,
        step3_output_path_b=args.step3_output_path_b,
        scene_idx_a=args.scene_idx_a,
        scene_idx_b=args.scene_idx_b,
        output=args.output,
        n_render_steps=args.n_render_steps,
        visual_only=args.visual_only,
        scene_b_offset=(args.offset_x, args.offset_y, args.offset_z),
    )
    vis.run()


if __name__ == "__main__":
    main()