"""
Copyright 2025 Zordi, Inc. All rights reserved.

Client for PA Arm Sim prediction service.
"""

import base64
from typing import List, Tuple

import cv2
import numpy as np
import requests

IMAGE_SIZE = (96, 96)
PAD_RATIO = 0.0


def process_image(
    image: np.ndarray,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    pad_ratio: float = PAD_RATIO,
) -> np.ndarray:
    """Perform center crop with padding based on the shorter dimension.

    Args:
        image: Input image as numpy array
        image_size: Tuple of (height, width)
        pad_ratio: Ratio to determine padding size (e.g., 0.2 for 20% padding)

    Returns:
        Cropped image as numpy array
    """
    height, width = image.shape[:2]
    shorter_dim = min(width, height)
    crop_size = int(shorter_dim * (1.0 - pad_ratio))

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    resized = cv2.resize(cropped, image_size)
    return resized


def encode_image(image: np.ndarray) -> str:
    """Encode image to base64 string.

    Args:
        image: Image array of shape (3, 96, 96)

    Returns:
        str: Base64 encoded image string
    """
    if image.ndim != 3:  # noqa: PLR2004
        raise ValueError("Color image must have 3 dimensions")

    if image.shape[0] == 3:  # noqa: PLR2004
        # Convert to numpy and rearrange dimensions
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    # Directly encode cv2/numpy array to base64
    _, buffer = cv2.imencode(".png", image)
    img_str = base64.b64encode(buffer.tobytes()).decode()

    return img_str


class PredictionClient:
    """Client for PA Arm Sim prediction service."""

    def __init__(self, host: str = "localhost", port: int = 10012):
        """Initialize client.

        Args:
            host: Server hostname
            port: Server port
        """
        self.base_url = f"http://{host}:{port}"

    def predict(
        self, images: List[np.ndarray], joint_positions: np.ndarray
    ) -> np.ndarray:
        """Get prediction from server."""
        images = [process_image(img) for img in images]

        # Prepare request data
        request_data = {
            "images": [encode_image(img) for img in images],
            "joint_positions": joint_positions.tolist(),
        }

        # Send request
        try:
            response = requests.post(f"{self.base_url}/predict", json=request_data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to server: {e!s}") from e

        # Parse response
        try:
            result = response.json()
            actions = np.array(result["actions"])
            return actions
        except Exception as e:
            raise ValueError(f"Failed to parse server response: {e!s}") from e


def main():
    """Example usage of PredictionClient."""
    # Create client
    client = PredictionClient()
    rng = np.random.default_rng(42)

    # Create dummy data - random uint8 images with values 0-255
    images = [
        rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(2)
    ]
    joint_positions = rng.standard_normal((2, 4))

    # Get prediction
    try:
        actions = client.predict(images, joint_positions)
        print(f"Received actions shape: {actions.shape}")
        print(f"Actions:\n{actions}")
    except Exception as e:
        print(f"Prediction failed: {e!s}")


if __name__ == "__main__":
    main()
