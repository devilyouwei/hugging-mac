"""Public API for the hugging-mac model SDK."""

from hugging_mac_sdk.converters import ConversionService, ConverterRegistry, ModelConverter
from hugging_mac_sdk.core.facade import ModelHandle, ModelSdk
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
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
from hugging_mac_sdk.schemas.catalog import (
    InstanceSnapshot,
    ModelCatalogSnapshot,
    ModelSummary,
    RuntimeSummary,
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
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
    UrlArchiveSource,
    UrlFileSource,
)

__all__ = [
    "ArtifactFormat",
    "BaseModelInstance",
    "BoundingBox",
    "ConversionRequest",
    "ConversionResult",
    "ConversionService",
    "ConverterRegistry",
    "Detection",
    "DetectionRequest",
    "DetectionResponse",
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
    "ManifestError",
    "ModelArtifactStatus",
    "ModelCatalogSnapshot",
    "ModelConverter",
    "ModelDefinition",
    "ModelHandle",
    "ModelLoadError",
    "ModelManifest",
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
    "RuntimePolicy",
    "RuntimeSpec",
    "RuntimeSummary",
    "UnsupportedCapabilityError",
    "UnsupportedRuntimeError",
    "UrlArchiveSource",
    "UrlFileSource",
]
