# GLM-OCR model package

Implements the MLX Community eight-bit checkpoint as the runtime-neutral
`DocumentParsing` capability. The package owns download, validation, serialized
MLX inference, streaming generation, and cleanup. Requests use GLM-OCR's native
`Text Recognition:` instruction internally because its supported prompts are limited.
