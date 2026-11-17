"""Deterministic media fixtures; preserve original sizes and sampling metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def cases() -> list[dict[str, Any]]:
    red = Image.new("RGB", (320, 256), (240, 20, 20))
    blue = Image.new("RGB", (256, 384), (20, 20, 240))
    color = {
        "type": "choice",
        "instructions": "What is the dominant color in the image?",
        "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
    }
    frames = np.stack([np.asarray(red)] * 4)
    return [
        {
            "id": "image-landscape",
            "payload": {
                "state": "Describe the image color.",
                "questions": {"color": color},
                "images": [red],
            },
        },
        {
            "id": "image-portrait",
            "payload": {
                "state": "Describe the image color.",
                "questions": {"color": color},
                "images": [blue],
            },
        },
        {
            "id": "two-images-multi-question",
            "payload": {
                "state": "Compare the two images.",
                "questions": {
                    "different": {
                        "type": "noul",
                        "instructions": "Are the two images different colors?",
                    },
                    "red": {
                        "type": "score",
                        "instructions": "How strongly is red present?",
                        "criteria": ["Absent", "Present"],
                    },
                },
                "images": [red, blue],
            },
        },
        {
            "id": "video-frames",
            "payload": {
                "state": "Describe the video color.",
                "questions": {
                    "color": {**color, "instructions": "What color dominates the video?"}
                },
                "videos": [frames],
            },
        },
        {
            "id": "image-and-video",
            "payload": {
                "state": "Compare the image and the video.",
                "questions": {
                    "different": {
                        "type": "noul",
                        "instructions": "Are the image and video different colors?",
                    }
                },
                "images": [blue],
                "videos": [frames],
            },
        },
    ]
