# Unlimited-OCR model package

Implements the MLX Community four-bit checkpoint as a runtime-neutral
`DocumentParsing` capability. The package owns checkpoint download, local
validation, serialized MLX-VLM inference, incremental text generation, and cleanup.

Requests support `gundam` (cropped 640-pixel views) and `base` (uncropped
1024-pixel views). Execution uses MLX on the Apple Silicon GPU.
