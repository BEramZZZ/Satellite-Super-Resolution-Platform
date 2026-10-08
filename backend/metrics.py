from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import torch
import lpips

from PIL import Image

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# DEVICE
# ============================================================

# Keep metric computation on CPU.
#
# Your RTX 2060 has 6 GB VRAM and the SR models should keep
# priority over metric models.
METRIC_DEVICE = torch.device("cpu")


# ============================================================
# LPIPS CACHE
# ============================================================

_LPIPS_MODEL = None


def get_lpips():
    """
    Load LPIPS once and keep it on CPU.
    """

    global _LPIPS_MODEL

    if _LPIPS_MODEL is None:

        print("[METRICS] Loading LPIPS Alex model on CPU...")

        _LPIPS_MODEL = (
            lpips.LPIPS(net="alex")
            .eval()
            .to(METRIC_DEVICE)
        )

        for parameter in _LPIPS_MODEL.parameters():
            parameter.requires_grad_(False)

        print("[METRICS] LPIPS loaded.")

    return _LPIPS_MODEL


# ============================================================
# IMAGE PREPARATION
# ============================================================

def _prepare_image(
    image,
    name: str,
) -> np.ndarray:
    """
    Normalize an image to:

        H x W x 3
        float32
        range [0, 1]

    Accepts either a PIL image (what main.py passes) or a
    numpy array.
    """

    # --------------------------------------------------------
    # PIL -> numpy (RGB)
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        image = np.asarray(
            image.convert("RGB")
        )

    if not isinstance(image, np.ndarray):

        raise TypeError(
            f"{name} must be a numpy.ndarray or PIL.Image."
        )

    # --------------------------------------------------------
    # Grayscale -> RGB
    # --------------------------------------------------------

    if image.ndim == 2:

        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image.ndim != 3:

        raise ValueError(
            f"{name} must have shape HxWxC. "
            f"Received {image.shape}."
        )

    # --------------------------------------------------------
    # RGBA -> RGB
    # --------------------------------------------------------

    if image.shape[2] == 4:

        image = image[:, :, :3]

    # --------------------------------------------------------
    # RGB only
    # --------------------------------------------------------

    if image.shape[2] != 3:

        raise ValueError(
            f"{name} must have 3 channels. "
            f"Received {image.shape[2]}."
        )

    # --------------------------------------------------------
    # float32
    # --------------------------------------------------------

    image = image.astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # [0,255] -> [0,1]
    # --------------------------------------------------------

    if image.max() > 1.0:

        image = image / 255.0

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


# ============================================================
# METRIC COMPUTATION
# ============================================================

@torch.no_grad()
def image_metrics(
    pred,
    target,
) -> Dict[str, float]:
    """
    Compute:

        PSNR
        SSIM
        LPIPS

    Accepted input:

        PIL image or numpy array, RGB
    """

    pred = _prepare_image(
        pred,
        "pred",
    )

    target = _prepare_image(
        target,
        "target",
    )

    # ========================================================
    # EXACT SHAPE MATCH
    # ========================================================

    if pred.shape != target.shape:

        raise ValueError(
            "Prediction and reference dimensions do not match: "
            f"pred={pred.shape}, "
            f"target={target.shape}"
        )

    # ========================================================
    # PSNR
    # ========================================================

    psnr = peak_signal_noise_ratio(
        target,
        pred,
        data_range=1.0,
    )

    # ========================================================
    # SSIM
    # ========================================================

    ssim = structural_similarity(
        target,
        pred,
        data_range=1.0,
        channel_axis=2,
    )

    # ========================================================
    # LPIPS
    # ========================================================

    # NumPy:
    #
    #     H x W x C
    #
    # LPIPS:
    #
    #     N x C x H x W
    #

    pred_tensor = torch.from_numpy(
        pred
    )

    target_tensor = torch.from_numpy(
        target
    )

    pred_tensor = (
        pred_tensor
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
        .to(
            METRIC_DEVICE,
            dtype=torch.float32,
        )
    )

    target_tensor = (
        target_tensor
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
        .to(
            METRIC_DEVICE,
            dtype=torch.float32,
        )
    )

    # LPIPS expects [-1, +1].
    pred_tensor = (
        pred_tensor * 2.0
    ) - 1.0

    target_tensor = (
        target_tensor * 2.0
    ) - 1.0

    lpips_model = get_lpips()

    lpips_value = float(
        lpips_model(
            pred_tensor,
            target_tensor,
        )
        .mean()
        .item()
    )

    del pred_tensor
    del target_tensor

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "PSNR": float(psnr),
        "SSIM": float(ssim),
        "LPIPS": lpips_value,
    }


# ============================================================
# SAFE METRIC WRAPPER
# ============================================================

def try_image_metrics(
    pred,
    target,
) -> Tuple[
    Optional[Dict[str, float]],
    Optional[str],
]:
    """
    Never lets a metric error destroy the SR output.

    Returns:

        (metrics, None)

    or:

        (None, error_message)
    """

    try:

        metrics = image_metrics(
            pred,
            target,
        )

        return metrics, None

    except Exception as exc:

        error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "[METRICS ERROR]",
            error,
        )

        return None, error