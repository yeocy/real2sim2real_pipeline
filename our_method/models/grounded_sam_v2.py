import torch.nn
import digital_cousins
import groundingdino
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# There seems to be an issue with torch's flash attention compatibility
# (https://github.com/facebookresearch/segment-anything-2/issues/48)
# So we implement the suggested hotfix here
import sam2.modeling.sam.transformer as transformer
import GPUtil
import cv2
transformer.OLD_GPU = True
transformer.USE_FLASH_ATTN = True
transformer.MATH_KERNEL_ON = True


GDINO_CHECKPOINT_PATH = f"{digital_cousins.CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
GSAMV2_CHECKPOINT_PATH = f"{digital_cousins.CHECKPOINT_DIR}/sam2_hiera_large.pt"
GSAMV2_CONFIG = "sam2_hiera_l.yaml"

# GSAMV2_CHECKPOINT_PATH = f"{digital_cousins.CHECKPOINT_DIR}/sam2_hiera_small.pt"
# GSAMV2_CONFIG = "sam2_hiera_s.yaml"

# GSAMV2_CHECKPOINT_PATH = f"{digital_cousins.CHECKPOINT_DIR}/sam2_hiera_tiny.pt"
# GSAMV2_CONFIG = "sam2_hiera_t.yaml"

class GroundedSAMv2(torch.nn.Module):

    def __init__(
        self,
        gdino=None,
        box_threshold=0.3,
        # box_threshold=0.01,
        text_threshold=0.25,
        # text_threshold=0.01,
        device="cuda",
    ):

        super().__init__()

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if gdino is None:
            gdino = load_model(
                f"{groundingdino.__path__[0]}/config/GroundingDINO_SwinT_OGC.py",
                GDINO_CHECKPOINT_PATH,
            )
        self.gdino = gdino
        self.gdino.eval()
        self.gdino.to(self.device)
        gpus = GPUtil.getGPUs()
        # for gpu in gpus:
        #     print(f"GPU {gpu.id}: {gpu.name}")
        #     print(f"  Used VRAM: {gpu.memoryUsed} MB")
        #     print(f"  Total VRAM: {gpu.memoryTotal} MB")
        #     print(f"  Usage: {gpu.memoryUsed/gpu.memoryTotal * 100:.2f}%")
        #     print("-" * 30)
        sam2 = build_sam2(GSAMV2_CONFIG, GSAMV2_CHECKPOINT_PATH, device=self.device)
        self.gsamv2 = SAM2ImagePredictor(sam2)
        gpus = GPUtil.getGPUs()
        # for gpu in gpus:
        #     print(f"GPU {gpu.id}: {gpu.name}")
        #     print(f"  Used VRAM: {gpu.memoryUsed} MB")
        #     print(f"  Total VRAM: {gpu.memoryTotal} MB")
        #     print(f"  Usage: {gpu.memoryUsed/gpu.memoryTotal * 100:.2f}%")
        #     print("-" * 30)

    def load_image(self, image_path):
        return load_image(image_path)

    def predict_boxes(self, img, caption):
        # Returns boxes, logits, phrases
        return predict(
            model=self.gdino,
            image=img,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

    def predict_segmentation(self, img_source, boxes, cxcywh=True, multimask_output=False):
        # 원본 이미지 크기 출력
        # print(f"Original image shape: {img_source.shape}")  # (907, 1612, 3)
        
        # # 이미지 크기 조정 (예: 50% 크기로 축소)
        # scale_percent = 10  # 크기를 50%로 줄이기
        # width = int(img_source.shape[1] * scale_percent / 100)
        # height = int(img_source.shape[0] * scale_percent / 100)
        # resized_img = cv2.resize(img_source, (width, height), interpolation=cv2.INTER_AREA)
        
        # print(f"Resized image shape: {resized_img.shape}")  # 조정된 이미지 크기 확인

        self.gsamv2.set_image(np.array(img_source))
        H, W, _ = img_source.shape

        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) if cxcywh else boxes
        boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H])

        
        # gpus = GPUtil.getGPUs()
        # for gpu in gpus:
        #     print(f"GPU {gpu.id}: {gpu.name}")
        #     print(f"  Used VRAM: {gpu.memoryUsed} MB")
        #     print(f"  Total VRAM: {gpu.memoryTotal} MB")
        #     print(f"  Usage: {gpu.memoryUsed/gpu.memoryTotal * 100:.2f}%")
        #     print("-" * 30)

        # torch.save(boxes_xyxy, "boxes.pt")

        masks, _, _ = self.gsamv2.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=multimask_output,
        )

        # Make sure masks is always shape 4
        if len(masks.shape) == 3:
            masks = masks.reshape(1, *masks.shape)

        return masks.astype(bool)
