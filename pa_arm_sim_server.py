"""
Copyright 2025 Zordi, Inc. All rights reserved.

REST API server for PA Arm Sim prediction model.
"""

import base64
import io
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pa_arm_sim_prediction import ActionModel
from PIL import Image
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    images: List[str]  # Base64 encoded images
    joint_positions: List[List[float]]  # List of joint positions


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    actions: List[List[float]]


app = FastAPI()
model: Optional[ActionModel] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model
    # TODO: Make this configurable
    model = ActionModel("checkpoint_pa_arm_sim_2025_0217.ckpt")


def decode_image(base64_str: str) -> torch.Tensor:
    """Decode base64 image string to torch tensor.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        torch.Tensor: Image tensor of shape (3, 96, 96)
    """
    try:
        # Decode base64 string to image
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))

        # Resize and convert to tensor
        img = img.resize((96, 96))
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float()

        # Rearrange dimensions from (H, W, C) to (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)

        # Normalize to [0, 1]
        img_tensor = img_tensor / 255.0

        return img_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e!s}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> Dict:
    """Predict actions from images and joint positions.

    Args:
        request: PredictRequest containing images and joint positions

    Returns:
        Dict containing predicted actions
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Convert images to tensor
        if len(request.images) != 2:
            raise HTTPException(status_code=400, detail="Exactly 2 images required")

        images = torch.stack([decode_image(img) for img in request.images])

        # Convert joint positions to tensor
        if len(request.joint_positions) != 2:
            raise HTTPException(
                status_code=400, detail="Exactly 2 joint positions required"
            )

        joint_positions = torch.tensor(request.joint_positions, dtype=torch.float32)

        # Get prediction
        actions = model.predict(images, joint_positions)

        # Convert to list for JSON response
        actions_list = actions.tolist()

        return {"actions": actions_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=10012)
