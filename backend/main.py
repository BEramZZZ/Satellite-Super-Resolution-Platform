from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

import torch
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# =============================================================================
# ESRGAN
# =============================================================================

from esrgan_model import (
    infer_esrgan,
    get_runtime_info as esrgan_runtime_info,
)


# =============================================================================
# ESRGAN MODEL REGISTRY
# =============================================================================

from model_registry import (
    get_model_info as get_esrgan_model_info,
    get_checkpoint as get_esrgan_checkpoint,
    is_available as is_esrgan_available,
    list_models as list_esrgan_models,
)


# =============================================================================
# DIFFUSION
# =============================================================================

from diffusion import (
    diffusion_availability,
    diffusion_runtime_info,
    image_to_png_bytes,
    run_diffusion,
)


# =============================================================================
# OPTIONAL METRICS
# =============================================================================
#
# Metrics are optional because inference should still work if the metric
# implementation is unavailable or a metric calculation fails.
#

try:
    from metrics import try_image_metrics
except ImportError:
    try_image_metrics = None


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Satellite Super-Resolution Backend",
    version="0.3.1",
    description=(
        "FastAPI backend for the Satellite Super-Resolution "
        "research/demo website."
    ),
)


# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        # Common
        "X-SR-Model",
        "X-SR-Strategy",
        "X-SR-Scale",
        "X-SR-Device",

        # Metrics
        "X-SR-Metrics-Available",
        "X-SR-PSNR",
        "X-SR-SSIM",
        "X-SR-LPIPS",
        "X-SR-Metrics-Error",

        # Diffusion
        "X-SR-Diffusion-Steps",
        "X-SR-Diffusion-Noise-Level",
        "X-SR-Diffusion-Guidance",
        "X-SR-Diffusion-Sampler",
        "X-SR-Diffusion-Eta",
    ],
)


# =============================================================================
# CONSTANTS
# =============================================================================

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

MAX_INPUT_DIMENSION = 1024


# =============================================================================
# BASIC ROUTES
# =============================================================================

@app.get("/")
def root():
    return {
        "service": "satellite-super-resolution-backend",
        "status": "ok",
        "version": "0.3.1",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "satellite-super-resolution-backend",
        "version": "0.3.1",
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "cuda_available": torch.cuda.is_available(),
    }


# =============================================================================
# ESRGAN MODEL AVAILABILITY
# =============================================================================

@app.get("/api/models/esrgan")
def esrgan_models():
    """
    Return the actual ESRGAN strategy registry.
    """

    return {
        "model": "Real-ESRGAN",
        "strategies": list_esrgan_models(),
        "runtime": esrgan_runtime_info(),
    }


# =============================================================================
# DIFFUSION MODEL AVAILABILITY
# =============================================================================

@app.get("/api/models/diffusion")
def diffusion_models():
    """
    Return the actual Diffusion runtime state.
    """

    return {
        "model": "stabilityai/stable-diffusion-x4-upscaler",
        "runtime": diffusion_runtime_info(),
        "availability": diffusion_availability(),
    }


# =============================================================================
# COMBINED MODEL AVAILABILITY
# =============================================================================

@app.get("/api/models")
def models():
    """
    Combined model availability endpoint used by the frontend.
    """

    return {
        "esrgan": {
            "model": "Real-ESRGAN",
            "strategies": list_esrgan_models(),
            "runtime": esrgan_runtime_info(),
        },
        "diffusion": {
            "model": "stabilityai/stable-diffusion-x4-upscaler",
            "availability": diffusion_availability(),
        },
    }


# =============================================================================
# IMAGE VALIDATION
# =============================================================================

def _validate_upload_metadata(
    file: UploadFile,
) -> None:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No image file was provided.",
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Supported formats: PNG, JPEG, JPG, WEBP."
            ),
        )


def _read_uploaded_image(
    file: UploadFile,
) -> Image.Image:
    """
    Read upload and decode it as RGB PIL image.
    """

    try:

        contents = file.file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum allowed size is "
                f"{MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    try:

        image = Image.open(
            BytesIO(contents)
        )

        image.load()

        return image.convert("RGB")

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {exc}",
        ) from exc


def _validate_dimensions(
    image: Image.Image,
    model_name: str,
) -> None:

    width, height = image.size

    if width < 1 or height < 1:

        raise HTTPException(
            status_code=400,
            detail="Invalid image dimensions.",
        )

    if (
        width > MAX_INPUT_DIMENSION
        or height > MAX_INPUT_DIMENSION
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"{model_name} interactive inference accepts "
                f"images up to {MAX_INPUT_DIMENSION}px per dimension. "
                f"Received {width}x{height}."
            ),
        )


def _read_reference_image(
    reference_file: UploadFile,
    lr_width: int,
    lr_height: int,
) -> Image.Image:
    """
    Read the optional HR reference and make sure it matches 4x scaling.
    """

    _validate_upload_metadata(
        reference_file
    )

    reference_image = _read_uploaded_image(
        reference_file
    )

    expected_reference_size = (
        lr_width * 4,
        lr_height * 4,
    )

    if reference_image.size != expected_reference_size:

        raise HTTPException(
            status_code=400,
            detail=(
                "HR reference dimensions must match 4x scaling. "
                f"Expected "
                f"{expected_reference_size[0]}x"
                f"{expected_reference_size[1]}, "
                f"got "
                f"{reference_image.size[0]}x"
                f"{reference_image.size[1]}."
            ),
        )

    return reference_image


# =============================================================================
# SHARED METRICS HELPER
# =============================================================================

def _compute_metrics(
    sr_image: Image.Image,
    reference_image: Optional[Image.Image],
) -> dict:
    """
    Compute PSNR / SSIM / LPIPS between the SR output and the HR reference.

    Never raises: a metric failure must not destroy a valid SR result.
    """

    result = {
        "available": False,
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "error": None,
    }

    if reference_image is None:

        return result

    if try_image_metrics is None:

        result["error"] = (
            "metrics.py could not be imported."
        )

        return result

    try:

        metric_result = try_image_metrics(
            sr_image,
            reference_image,
        )

        # try_image_metrics returns (metrics_dict, error_message).
        # Also tolerate a bare metrics dict.

        if isinstance(
            metric_result,
            tuple,
        ):

            metrics_dict = metric_result[0]

            if (
                len(metric_result) > 1
                and metric_result[1] is not None
            ):

                result["error"] = str(
                    metric_result[1]
                )

        else:

            metrics_dict = metric_result

        if isinstance(
            metrics_dict,
            dict,
        ):

            result["PSNR"] = metrics_dict.get(
                "PSNR",
                metrics_dict.get("psnr"),
            )

            result["SSIM"] = metrics_dict.get(
                "SSIM",
                metrics_dict.get("ssim"),
            )

            result["LPIPS"] = metrics_dict.get(
                "LPIPS",
                metrics_dict.get("lpips"),
            )

            result["available"] = any(
                result[key] is not None
                for key in (
                    "PSNR",
                    "SSIM",
                    "LPIPS",
                )
            )

    except Exception as exc:

        result["available"] = False
        result["error"] = str(exc)

    return result


def _apply_metric_headers(
    response: StreamingResponse,
    metrics: dict,
) -> None:
    """
    Write the metric result into the X-SR-* response headers.
    """

    response.headers[
        "X-SR-Metrics-Available"
    ] = (
        "true"
        if metrics["available"]
        else "false"
    )

    if metrics["PSNR"] is not None:

        response.headers[
            "X-SR-PSNR"
        ] = str(
            float(metrics["PSNR"])
        )

    if metrics["SSIM"] is not None:

        response.headers[
            "X-SR-SSIM"
        ] = str(
            float(metrics["SSIM"])
        )

    if metrics["LPIPS"] is not None:

        response.headers[
            "X-SR-LPIPS"
        ] = str(
            float(metrics["LPIPS"])
        )

    if metrics["error"]:

        response.headers[
            "X-SR-Metrics-Error"
        ] = str(metrics["error"])[:1000]


# =============================================================================
# ESRGAN STRATEGY NORMALIZATION
# =============================================================================

def _normalize_esrgan_strategy(
    strategy: str,
) -> str:
    """
    Accept the different spellings that may come from the frontend,
    then convert them to the exact names used by model_registry.py /
    esrgan_model.py.
    """

    strategy = strategy.strip().lower()

    aliases = {
        "zero": "zero-shot",
        "zero_shot": "zero-shot",
        "zero-shot": "zero-shot",

        "head": "head-only",
        "headonly": "head-only",
        "head_only": "head-only",
        "head-only": "head-only",

        "partial": "partial",

        "lora": "lora",

        "full": "full",
    }

    return aliases.get(
        strategy,
        strategy,
    )


# =============================================================================
# ESRGAN INFERENCE
# =============================================================================

@app.post("/api/inference/esrgan")
def esrgan_inference(
    file: UploadFile = File(...),
    strategy: str = Form("zero-shot"),
    reference_file: Optional[UploadFile] = File(None),
):
    """
    Real-ESRGAN 4x super-resolution.

    Strategies:
        zero-shot
        head-only
        partial
        lora
        full

    Optional:
        reference_file -> HR reference for PSNR / SSIM / LPIPS
    """

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    strategy = _normalize_esrgan_strategy(
        strategy
    )

    allowed_strategies = [
        "zero-shot",
        "head-only",
        "partial",
        "lora",
        "full",
    ]

    if strategy not in allowed_strategies:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Unknown ESRGAN strategy: {strategy}"
                ),
                "supported_strategies": allowed_strategies,
            },
        )

    # -------------------------------------------------------------------------
    # Upload
    # -------------------------------------------------------------------------

    _validate_upload_metadata(file)

    lr_image = _read_uploaded_image(file)

    _validate_dimensions(
        lr_image,
        "ESRGAN",
    )

    lr_width, lr_height = lr_image.size

    # -------------------------------------------------------------------------
    # Checkpoint availability
    # -------------------------------------------------------------------------

    try:

        available = is_esrgan_available(
            strategy
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not inspect ESRGAN strategy "
                f"'{strategy}': {exc}"
            ),
        ) from exc

    if not available:

        try:
            model_info = get_esrgan_model_info(
                strategy
            )
        except Exception:
            model_info = {
                "strategy": strategy,
                "available": False,
            }

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"ESRGAN strategy '{strategy}' "
                    "is currently unavailable."
                ),
                "model": model_info,
            },
        )

    # -------------------------------------------------------------------------
    # Optional HR reference
    # -------------------------------------------------------------------------

    reference_image: Optional[Image.Image] = None

    if reference_file is not None:

        reference_image = _read_reference_image(
            reference_file,
            lr_width,
            lr_height,
        )

    # -------------------------------------------------------------------------
    # Actual ESRGAN inference
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    # Your existing module exposes infer_esrgan(), not run_esrgan().
    #

    try:

        sr_image = infer_esrgan(
            lr_image,
            strategy,
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        raise HTTPException(
            status_code=507,
            detail=(
                "CUDA out of memory during ESRGAN inference. "
                "Try a smaller LR image."
            ),
        ) from exc

    except RuntimeError as exc:

        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"ESRGAN runtime error: {exc}",
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"ESRGAN inference failed: {exc}",
        ) from exc

    # -------------------------------------------------------------------------
    # Verify result
    # -------------------------------------------------------------------------

    if not isinstance(
        sr_image,
        Image.Image,
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "ESRGAN inference did not return a PIL image."
            ),
        )

    sr_image = sr_image.convert("RGB")

    expected_size = (
        lr_width * 4,
        lr_height * 4,
    )

    if sr_image.size != expected_size:

        raise HTTPException(
            status_code=500,
            detail=(
                "ESRGAN returned an unexpected output size. "
                f"Expected "
                f"{expected_size[0]}x{expected_size[1]}, "
                f"got "
                f"{sr_image.size[0]}x{sr_image.size[1]}."
            ),
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = _compute_metrics(
        sr_image,
        reference_image,
    )

    # -------------------------------------------------------------------------
    # Serialize
    # -------------------------------------------------------------------------

    try:

        output_bytes = image_to_png_bytes(
            sr_image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not serialize ESRGAN output: {exc}"
            ),
        ) from exc

    response = StreamingResponse(
        BytesIO(output_bytes),
        media_type="image/png",
    )

    # -------------------------------------------------------------------------
    # Response headers
    # -------------------------------------------------------------------------

    response.headers["X-SR-Model"] = (
        "Real-ESRGAN"
    )

    response.headers["X-SR-Strategy"] = (
        strategy
    )

    response.headers["X-SR-Scale"] = "4"

    response.headers["X-SR-Device"] = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _apply_metric_headers(
        response,
        metrics,
    )

    return response


# =============================================================================
# DIFFUSION INFERENCE
# =============================================================================

@app.post("/api/inference/diffusion")
def diffusion_inference(
    file: UploadFile = File(...),
    strategy: str = Form("zero_shot"),
    reference_file: Optional[UploadFile] = File(None),
):
    """
    Stable Diffusion x4 Upscaler.

    The exact sampling configuration lives in diffusion.py.

    Optional:
        reference_file -> HR reference for PSNR / SSIM / LPIPS
    """

    strategy = strategy.strip().lower()

    # -------------------------------------------------------------------------
    # Upload
    # -------------------------------------------------------------------------

    _validate_upload_metadata(file)

    lr_image = _read_uploaded_image(file)

    _validate_dimensions(
        lr_image,
        "Diffusion",
    )

    lr_width, lr_height = lr_image.size

    # -------------------------------------------------------------------------
    # Optional HR reference
    # -------------------------------------------------------------------------

    reference_image: Optional[Image.Image] = None

    if reference_file is not None:

        reference_image = _read_reference_image(
            reference_file,
            lr_width,
            lr_height,
        )

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    availability = diffusion_availability()

    if strategy not in availability:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Unknown diffusion strategy: {strategy}"
                ),
                "supported_strategies": list(
                    availability.keys()
                ),
            },
        )

    strategy_info = availability[
        strategy
    ]

    if not strategy_info["available"]:

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"Diffusion strategy '{strategy}' "
                    "is currently unavailable."
                ),
                "reason": strategy_info["reason"],
                "checkpoint": strategy_info["checkpoint"],
            },
        )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    try:

        sr_image = run_diffusion(
            lr_image=lr_image,
            strategy=strategy,
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        raise HTTPException(
            status_code=507,
            detail=(
                "CUDA out of memory during Diffusion inference. "
                "Try a smaller LR image."
            ),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Diffusion runtime error: {exc}",
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Diffusion inference failed: {exc}"
            ),
        ) from exc

    # -------------------------------------------------------------------------
    # Verify output
    # -------------------------------------------------------------------------

    sr_image = sr_image.convert("RGB")

    expected_size = (
        lr_width * 4,
        lr_height * 4,
    )

    if sr_image.size != expected_size:

        raise HTTPException(
            status_code=500,
            detail=(
                "Diffusion returned an unexpected output size. "
                f"Expected "
                f"{expected_size[0]}x{expected_size[1]}, "
                f"got "
                f"{sr_image.size[0]}x{sr_image.size[1]}."
            ),
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = _compute_metrics(
        sr_image,
        reference_image,
    )

    # -------------------------------------------------------------------------
    # Serialize
    # -------------------------------------------------------------------------

    try:

        output_bytes = image_to_png_bytes(
            sr_image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not serialize Diffusion output: {exc}"
            ),
        ) from exc

    response = StreamingResponse(
        BytesIO(output_bytes),
        media_type="image/png",
    )

    # -------------------------------------------------------------------------
    # Headers
    # -------------------------------------------------------------------------

    response.headers["X-SR-Model"] = (
        "stabilityai/stable-diffusion-x4-upscaler"
    )

    response.headers["X-SR-Strategy"] = (
        strategy
    )

    response.headers["X-SR-Scale"] = "4"

    response.headers["X-SR-Device"] = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    response.headers[
        "X-SR-Diffusion-Steps"
    ] = "30"

    response.headers[
        "X-SR-Diffusion-Noise-Level"
    ] = "20"

    response.headers[
        "X-SR-Diffusion-Guidance"
    ] = "9.0"

    response.headers[
        "X-SR-Diffusion-Sampler"
    ] = "DDIM"

    response.headers[
        "X-SR-Diffusion-Eta"
    ] = "0.0"

    _apply_metric_headers(
        response,
        metrics,
    )

    return response