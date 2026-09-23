"""Runtime backend contracts and registry."""

from hugging_mac_sdk.runtime.base import RuntimeAdapter, RuntimeBackend, RuntimeSession
from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider, CoreMLSession
from hugging_mac_sdk.runtime.device import DeviceInfo, DeviceKind
from hugging_mac_sdk.runtime.mlx import MlxProvider, MlxSession
from hugging_mac_sdk.runtime.onnx import OnnxRuntimeProvider, OnnxRuntimeSession
from hugging_mac_sdk.runtime.registry import RuntimeRegistry, create_default_runtime_registry
from hugging_mac_sdk.runtime.torch import TorchProvider, TorchSession

__all__ = [
    "CoreAIProvider",
    "CoreAISession",
    "CoreMLProvider",
    "CoreMLSession",
    "DeviceInfo",
    "DeviceKind",
    "MlxProvider",
    "MlxSession",
    "OnnxRuntimeProvider",
    "OnnxRuntimeSession",
    "RuntimeAdapter",
    "RuntimeBackend",
    "RuntimeRegistry",
    "RuntimeSession",
    "TorchProvider",
    "TorchSession",
    "create_default_runtime_registry",
]
