from email.mime import image
import numpy as np
from regex import P
import torch as th
import trimesh
import digital_cousins.utils.transform_utils as T
from digital_cousins.utils.processing_utils import distance_to_plane, create_polygon_from_vertices
import omnigibson as og
from copy import deepcopy
from our_method.utils.processing_utils import get_reproject_offset
from PIL import Image
from our_method.utils.processing_utils import NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from our_method.utils.dataset_utils import get_all_dataset_categories, get_all_articulated_categories, \
    extract_info_from_model_snapshot, ARTICULATION_INFO, ARTICULATION_VALID_ANGLES
import os

from digital_cousins.models.clip import CLIPEncoder
from digital_cousins.models.dino_v2 import DinoV2Encoder
import open3d as o3d
import cv2

def concat_images(snapshot_path, snapshot_list_files, visualize_resolution=(640, 480), images_per_row=10, fontscale = 2, save_path=None):
    """
    snapshot_list_files 내 이미지들을 한 줄에 10개씩 정렬하고, 왼쪽 위에 파일명 숫자 라벨을 추가.

    Args:
        snapshot_path (str): 이미지가 들어있는 디렉토리 경로
        snapshot_list_files (list): 이미지 파일 이름 리스트
        visualize_resolution (tuple): 각 이미지 리사이즈 크기
        images_per_row (int): 한 줄에 넣을 이미지 수
        save_path (str or None): 저장 경로. None이면 저장하지 않음

    Returns:
        np.array: 합쳐진 이미지 배열 (H_total, W_total, 3)
    """
    resized_labeled_imgs = []
    image_list = [name for name in snapshot_list_files if int(name.split('_')[1].split('.')[0]) % 25 == 0]

    for fname in image_list:
        img_path = os.path.join(snapshot_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[경고] 이미지를 불러올 수 없습니다: {img_path}")
            continue
        img_resized = cv2.resize(img, visualize_resolution)

        # 텍스트용 숫자 추출
        try:
            label = fname.split("_")[-1].split(".")[0]  # 예: 'jqyhjy_24.png' -> '24'
        except:
            label = "?"

        # 텍스트 추가
        cv2.putText(
            img_resized,
            label,
            org=(5, int(30 * fontscale)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontscale,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        resized_labeled_imgs.append(img_resized)

    # Grid로 연결
    rows = []
    for i in range(0, len(resized_labeled_imgs), images_per_row):
        row_imgs = resized_labeled_imgs[i:i + images_per_row]
        row = np.concatenate(row_imgs, axis=1)
        rows.append(row)
    full_img = np.concatenate(rows, axis=0)

    if save_path:
        Image.fromarray(full_img).save(save_path)
        print(f"[저장됨] {save_path}")

    return full_img

object_local_path = "/home/yeocy/robotics/LLMforMani/Simulation/digital-cousins/assets/objects"
category = "metal_bottom_cabinet"
model_id = "jqyhjy"

snapshot_path = f"{object_local_path}/{category}/model/{model_id}/"

# 해당 경로에 있는 파일들을 가져와 정렬된 리스트로 만들기
if os.path.exists(snapshot_path):
    snapshot_list_files = sorted(os.listdir(snapshot_path))
else:
    print(f"경로가 존재하지 않습니다: {snapshot_path}")

output_img = concat_images(
    snapshot_path=snapshot_path,
    snapshot_list_files=snapshot_list_files,
    visualize_resolution=(640, 480),  # 해상도 조절 가능
    images_per_row=10,
    save_path="snapshot_grid.png"
)
