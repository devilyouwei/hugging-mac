"""Public API for the hugging-mac model SDK."""

from hugging_mac_sdk.converters import ConversionService, ConverterRegistry, ModelConverter
from hugging_mac_sdk.core.config import ModelPackageConfig, load_model_config
from hugging_mac_sdk.core.facade import ModelHandle, ModelSdk
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelInstanceInfo, ModelState
from hugging_mac_sdk.core.manager import InstanceManager, ReusePolicy
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.core.resources import ModelResourceProvider, ModelResourceService
from hugging_mac_sdk.core.runtime_policy import RuntimePolicy
from hugging_mac_sdk.errors import (
    DownloadError,
    HuggingMacSdkError,
    InferenceError,
    InsufficientResourceError,
    ManifestError,
    ModelLoadError,
    RegistrationConflictError,
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedCapabilityError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources import DownloadProgress, ResourceDownloader
from hugging_mac_sdk.runtime import (
    DeviceInfo,
    DeviceKind,
    RuntimeAdapter,
    RuntimeBackend,
    RuntimeRegistry,
    RuntimeSession,
)
from hugging_mac_sdk.schemas.artifact import ArtifactKind, ModelArtifact
from hugging_mac_sdk.schemas.catalog import (
    InstanceSnapshot,
    LifecycleMetrics,
    LifecycleOperation,
    ModelCatalogSnapshot,
    ModelSummary,
    RuntimeSummary,
    UnloadResult,
)
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionRequest,
    DetectionResponse,
    ImageInput,
)
from hugging_mac_sdk.schemas.health import HealthReport, HealthStatus
from hugging_mac_sdk.schemas.manifest import ModelManifest, RuntimeSpec
from hugging_mac_sdk.schemas.resources import (
    ConversionTargetStatus,
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
    RuntimeResourceStatus,
    UrlArchiveSource,
    UrlFileSource,
)

__all__ = [
    "ArtifactFormat",
    "ArtifactKind",
    "BaseModelInstance",
    "BoundingBox",
    "ConversionRequest",
    "ConversionResult",
    "ConversionService",
    "ConversionTargetStatus",
    "ConverterRegistry",
    "Detection",
    "DetectionRequest",
    "DetectionResponse",
    "DeviceInfo",
    "DeviceKind",
    "DownloadError",
    "DownloadProgress",
    "HealthReport",
    "HealthStatus",
    "HuggingFaceSource",
    "HuggingMacSdkError",
    "ImageInput",
    "InferenceError",
    "InstanceManager",
    "InstanceSnapshot",
    "InsufficientResourceError",
    "LifecycleMetrics",
    "LifecycleOperation",
    "ManifestError",
    "ModelArtifact",
    "ModelArtifactStatus",
    "ModelCatalogSnapshot",
    "ModelConverter",
    "ModelDefinition",
    "ModelHandle",
    "ModelInstanceInfo",
    "ModelLoadError",
    "ModelManifest",
    "ModelPackageConfig",
    "ModelRegistry",
    "ModelResourceProvider",
    "ModelResourceService",
    "ModelResourceStatus",
    "ModelSdk",
    "ModelState",
    "ModelSummary",
    "RegistrationConflictError",
    "ResolvedResource",
    "ResourceDownloader",
    "ResourceIntegrityError",
    "ResourceNotFoundError",
    "ReusePolicy",
    "RuntimeAdapter",
    "RuntimeBackend",
    "RuntimePolicy",
    "RuntimeRegistry",
    "RuntimeResourceStatus",
    "RuntimeSession",
    "RuntimeSpec",
    "RuntimeSummary",
    "UnloadResult",
    "UnsupportedCapabilityError",
    "UnsupportedRuntimeError",
    "UrlArchiveSource",
    "UrlFileSource",
    "load_model_config",
]
