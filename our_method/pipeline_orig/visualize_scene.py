"""
Visualize a simulated scene from a step3_output_path JSON file.
"""
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch as th
import numpy as np
import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
import our_method.utils.transform_utils as T


class VisualizeScene:
    def __init__(self, step3_output_path, scene_idx=0, output=None, n_render_steps=100, visual_only=False):
        self.step3_output_path = step3_output_path
        self.scene_idx = scene_idx
        self.output = output
        self.n_render_steps = n_render_steps
        self.visual_only = visual_only

    def create_scene(self, floor=True, sky=True):
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    def load_scene(self, scene_info, visual_only=False):
        scene = self.create_scene(floor=True)
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(f)
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(
            th.tensor(cam_pose[0], dtype=th.float),
            th.tensor(cam_pose[1], dtype=th.float)
        )
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
                obj.set_position_orientation(
                    th.tensor(obj_pos, dtype=th.float),
                    th.tensor(obj_quat, dtype=th.float)
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
        with open(self.step3_output_path, "r") as f:
            step3_info = json.load(f)
        scene_key = f"scene_{self.scene_idx}"
        if scene_key not in step3_info:
            raise ValueError(f"Scene {scene_key} not found in {self.step3_output_path}")
        scene_info = step3_info[scene_key]

        og.launch()
        scene = self.load_scene(scene_info, visual_only=self.visual_only)
        scene_rgb = self.take_photo(n_render_steps=self.n_render_steps)

        output_path = self.output
        if output_path is None:
            out_dir = os.path.dirname(self.step3_output_path)
            output_path = os.path.join(out_dir, f"{scene_key}_visualization.png")
        Image.fromarray(scene_rgb).save(output_path)
        print(f"Scene visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3_output_path", type=str, required=True, help="Path to step_3_output_info.json")
    parser.add_argument("--scene_idx", type=int, default=0, help="Scene index to visualize (default: 0)")
    parser.add_argument("--output", type=str, default=None, help="Output image path (optional)")
    parser.add_argument("--n_render_steps", type=int, default=100, help="Number of render steps before capture")
    parser.add_argument("--visual_only", action="store_true", help="Load objects as visual only")
    args = parser.parse_args()

    vis = VisualizeScene(
        step3_output_path=args.step3_output_path,
        scene_idx=args.scene_idx,
        output=args.output,
        n_render_steps=args.n_render_steps,
        visual_only=args.visual_only,
    )
    vis.run()

if __name__ == "__main__":
    main()
