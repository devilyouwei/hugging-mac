# GOT-OCR2.0 model package

Implements the MLX Community eight-bit checkpoint as a runtime-neutral
`DocumentParsing` capability. The package owns checkpoint download, local
validation, serialized MLX-VLM inference, incremental text generation, and cleanup.

GOT-OCR2.0 only supports its native OCR instructions. The engine therefore maps
SDK document parsing requests to the model's formatted OCR instruction internally.
Ordered multi-image requests are processed one page at a time and returned with
Markdown page provenance markers.
