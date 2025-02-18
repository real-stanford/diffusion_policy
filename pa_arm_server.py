"""
Copyright 2025 Zordi, Inc. All rights reserved.

REST API server for PA Arm Sim prediction model.
"""

import base64
import io
import logging
import os
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pa_arm_sim_prediction import ActionModel
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT", "checkpoints/checkpoint_pa_arm_sim_2025_0217.ckpt"
)
IMAGE_SIZE = 96
REQUIRED_SEQUENCE_LENGTH = 2


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    images: List[str] = Field(
        min_length=REQUIRED_SEQUENCE_LENGTH, max_length=REQUIRED_SEQUENCE_LENGTH
    )
    joint_positions: List[List[float]] = Field(
        min_length=REQUIRED_SEQUENCE_LENGTH, max_length=REQUIRED_SEQUENCE_LENGTH
    )


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    actions: List[List[float]]


app = FastAPI()
model: Optional[ActionModel] = None

# Create transform pipeline once
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def decode_image(base64_str: str) -> torch.Tensor:
    """Decode base64 image string to torch tensor.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        torch.Tensor: Image tensor of shape (3, 96, 96)

    Raises:
        HTTPException: If image decoding fails
    """
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return image_transform(img)  # type: ignore[return-value]

    except Exception as e:
        logger.error("Image decoding failed: %s", e)
        raise HTTPException(
            status_code=400, detail="Failed to decode image: %s" % e
        ) from e


@app.on_event("startup")
def startup_event() -> None:
    """Initialize model on startup."""
    global model  # noqa: PLW0603
    try:
        logger.info("Loading model from %s", MODEL_CHECKPOINT)
        model = ActionModel(MODEL_CHECKPOINT)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise RuntimeError("Failed to initialize model: %s" % e) from e


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> Dict:
    """Predict actions from images and joint positions.

    Args:
        request: PredictRequest containing images and joint positions

    Returns:
        Dict containing predicted actions

    Raises:
        HTTPException: If prediction fails
    """
    if model is None:
        logger.error("Model not initialized")
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Convert images to tensor
        images = torch.stack([decode_image(img) for img in request.images])

        # Convert joint positions to tensor
        joint_positions = torch.tensor(request.joint_positions, dtype=torch.float32)

        # Validate joint position values
        if not torch.all(torch.isfinite(joint_positions)):
            raise ValueError("Joint positions contain invalid values")

        # Get prediction
        actions = model.predict(images, joint_positions)
        return {"actions": actions.tolist()}

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed: %s" % e) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=10012)
