import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from unidepth.models import UniDepthV2
from digital_cousins.utils.processing_utils import process_depth_linear


class UniDepthV2Wrapper(torch.nn.Module):
    """
    Lightweight wrapper for the UniDepth V2 model to perform depth map inference from images.
    """
    def __init__(
            self,
            device="cuda",
    ):
        """
        Initialize the UniDepthV2Wrapper.

        Args:
            device (str): Device to run the model on. Default is "cuda" if available, else "cpu".
        """
        super().__init__()
        # Load the UniDepth V2 model (ViT-L/14 backbone by default)
        self.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device).eval()

    def estimate_depth_and_intrinsic_linear(self, input_path, output_path, depth_limits=(0, 10.0)):
        """
        Estimate a linear depth map from an input image using the UniDepth V2 model.

        Args:
            input_path (str): Path to the input RGB image file.
            output_path (str): Path to save the resulting depth map image.
            depth_limits (tuple): (min, max) range for normalizing the depth map when saving.

        Returns:
            np.ndarray: The estimated linear depth map as a NumPy array.
        """
        # Load image and convert to tensor (C, H, W)
        img = torch.from_numpy(np.array(Image.open(input_path))).permute(2, 0, 1)
        # Run inference to get depth prediction (in meters)
        pred = self.model.infer(img)
        pred_depth = pred["depth"].squeeze().cpu().numpy()
        K = pred["intrinsics"].squeeze(0).cpu().numpy()   # (3, 3)

        # Ensure output directory exists
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        # Normalize and convert depth to 8-bit image for saving
        depth = process_depth_linear(depth=pred_depth, in_limits=depth_limits)
        Image.fromarray(depth).save(output_path)

        return K, depth