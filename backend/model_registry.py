from __future__ import annotations

from pathlib import Path


# ============================================================
# MODEL DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# ESRGAN MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "zero-shot": {
        "model": "Real-ESRGAN",
        "strategy": "zero-shot",
        "checkpoint": MODELS_DIR / "RealESRGAN_x4plus.pth",
        "scale": 4,
    },

    "head-only": {
        "model": "Real-ESRGAN",
        "strategy": "head-only",
        "checkpoint": MODELS_DIR / "esrgan_head_only.pth",
        "scale": 4,
    },

    "partial": {
        "model": "Real-ESRGAN",
        "strategy": "partial",
        "checkpoint": MODELS_DIR / "esrgan_partial.pth",
        "scale": 4,
    },

    "lora": {
        "model": "Real-ESRGAN",
        "strategy": "lora",
        "checkpoint": MODELS_DIR / "esrgan_lora.pth",
        "scale": 4,
    },

    "full": {
        "model": "Real-ESRGAN",
        "strategy": "full",
        "checkpoint": MODELS_DIR / "esrgan_full.pth",
        "scale": 4,
    },
}


# ============================================================
# ESRGAN STRATEGIES
# ============================================================

ESRGAN_STRATEGIES = [
    "zero-shot",
    "head-only",
    "partial",
    "lora",
    "full",
]


# ============================================================
# GET MODEL INFO
# ============================================================

def get_model_info(
    strategy: str,
) -> dict:

    if strategy not in MODEL_REGISTRY:

        raise ValueError(
            f"Unknown ESRGAN strategy: {strategy}"
        )

    config = MODEL_REGISTRY[
        strategy
    ]

    checkpoint = config[
        "checkpoint"
    ]

    return {
        "model": config["model"],
        "strategy": config["strategy"],
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "scale": config["scale"],
        "exists": checkpoint.exists(),
        "available": checkpoint.exists(),
    }


# ============================================================
# GET CHECKPOINT
# ============================================================

def get_checkpoint(
    strategy: str,
) -> Path:

    if strategy not in MODEL_REGISTRY:

        raise ValueError(
            f"Unknown ESRGAN strategy: {strategy}"
        )

    checkpoint = MODEL_REGISTRY[
        strategy
    ]["checkpoint"]

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    return checkpoint


# ============================================================
# LIST ALL MODELS
# ============================================================

def list_models() -> list[dict]:

    models = []

    for strategy in ESRGAN_STRATEGIES:

        config = MODEL_REGISTRY[
            strategy
        ]

        checkpoint = config[
            "checkpoint"
        ]

        models.append(
            {
                "model": config["model"],
                "strategy": config["strategy"],
                "checkpoint": checkpoint.name,
                "scale": config["scale"],
                "exists": checkpoint.exists(),
                "available": checkpoint.exists(),
            }
        )

    return models


# ============================================================
# CHECK WHETHER STRATEGY IS AVAILABLE
# ============================================================

def is_available(
    strategy: str,
) -> bool:

    if strategy not in MODEL_REGISTRY:
        return False

    return MODEL_REGISTRY[
        strategy
    ]["checkpoint"].exists()