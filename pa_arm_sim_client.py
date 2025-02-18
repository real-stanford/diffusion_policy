"""
Copyright 2025 Zordi, Inc. All rights reserved.

Client for PA Arm Sim prediction service.
"""

import base64
import io

import numpy as np
import requests
import torch
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

    def encode_image(self, image: torch.Tensor) -> str:
        """Encode image tensor to base64 string.

        Args:
            image: Image tensor of shape (3, 96, 96)

        Returns:
            str: Base64 encoded image string
        """
        # Convert to numpy and rearrange dimensions
        img_array = image.numpy()
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
        self, images: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Get prediction from server.

        Args:
            images: Image tensor of shape (2, 3, 96, 96)
            joint_positions: Joint positions tensor of shape (2, 4)

        Returns:
            torch.Tensor: Predicted actions of shape (8, 4)
        """
        # Validate input shapes
        if images.shape != (2, 3, 96, 96):
            raise ValueError(
                f"Expected images shape (2, 3, 96, 96), got {images.shape}"
            )
        if joint_positions.shape != (2, 4):
            raise ValueError(
                f"Expected joint_positions shape (2, 4), got {joint_positions.shape}"
            )

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
            actions = torch.tensor(result["actions"])
            return actions
        except Exception as e:
            raise ValueError(f"Failed to parse server response: {e!s}") from e


def main():
    """Example usage of PredictionClient."""
    # Create client
    client = PredictionClient()

    # Create dummy data
    images = torch.randn(2, 3, 96, 96)
    joint_positions = torch.randn(2, 4)

    # Get prediction
    try:
        actions = client.predict(images, joint_positions)
        print(f"Received actions shape: {actions.shape}")
        print(f"Actions:\n{actions}")
    except Exception as e:
        print(f"Prediction failed: {e!s}")


if __name__ == "__main__":
    main()
