"""Object detection inference providers."""

from vision.object_detection.providers.base import InferenceProvider, ObjectDetection
from vision.object_detection.providers.local import LocalInferenceProvider
from vision.object_detection.providers.node import NodeInferenceProvider

__all__ = [
    "InferenceProvider",
    "ObjectDetection",
    "LocalInferenceProvider",
    "NodeInferenceProvider",
]
