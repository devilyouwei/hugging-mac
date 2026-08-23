# Qwen3 TTS

## Purpose

Qwen3 TTS generates multilingual speech and supports voice cloning on the MLX
runtime.

## Support

- Model ID: `qwen/qwen3-tts-12hz`
- Variants: `0.6b-base` (default)
- Runtimes: `mlx` (default), `coreml`
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Platform: Apple Silicon macOS
- Optional dependency set: `tts`

## Prepare resources

Select a runtime while preparing resources. This example uses MLX.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.models.qwen3_tts import register_qwen3_tts


async def main() -> None:
    sdk = ModelSdk()
    register_qwen3_tts(sdk.registry)
    model_id = "qwen/qwen3-tts-12hz"

    await sdk.resources.download_source(
        model_id,
        variant="0.6b-base",
        options={"runtime": "mlx"},
    )
    async with await sdk.load(model_id, runtime="mlx") as handle:
        synthesizer = handle.require(SpeechSynthesis)
        result = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text="Hello from Qwen3 TTS.",
                language="english",
            )
        )
        Path("qwen3-tts.f32le").write_bytes(result.audio)


asyncio.run(main())
```

## Notes and limits

MLX can generate directly without a reference, or clone a voice when both
reference audio and its transcript are supplied. Reference parameters are
omitted entirely for direct generation. The Core ML runtime supports a fixed
voice and a more limited request set; unsupported reference or speed options are
rejected explicitly.

## License

The model is licensed under `Apache-2.0`.
