"""
Copyright 2025 Zordi, Inc. All rights reserved.

Client for PA Arm Sim prediction service.
"""

import base64
import io
from typing import List

import numpy as np
import requests
from PIL import Image


class PredictionClient:
    """Client for PA Arm Sim prediction service."""

    def __init__(self, host: str = "localhost", port: int = 10012):
        """Initialize client.

        Args:
            host: Server hostname
            port: Server port
        """
        self.base_url = f"http://{host}:{port}"

    def encode_image(self, img_array: np.ndarray) -> str:
        """Encode image to base64 string.

        Args:
            image: Image array of shape (3, 96, 96)

        Returns:
            str: Base64 encoded image string
        """
        # Convert to numpy and rearrange dimensions
        img_array = np.transpose(img_array, (1, 2, 0))

        # Scale to 0-255 and convert to uint8
        img_array = (img_array * 255).astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(img_array)

        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    def predict(
        self, images: List[np.ndarray], joint_positions: np.ndarray
    ) -> np.ndarray:
        """Get prediction from server."""
        # Prepare request data
        request_data = {
            "images": [self.encode_image(img) for img in images],
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

    # Create dummy data
    images = [rng.standard_normal((3, 96, 96)) for _ in range(2)]
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
