# Kokoro 82M

## Purpose

Kokoro is a compact text-to-speech model with selectable voices and speech speed.

## Support

- Model ID: `hexgrad/kokoro`
- Variants: `v1.0` (default)
- Runtimes: `pytorch-mps`, `coreml` (default)
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Platform: Core ML acceleration requires Apple Silicon
- Optional dependency set: `tts`

## Prepare resources

The example downloads and loads the default Core ML runtime.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.models.kokoro import register_kokoro


async def main() -> None:
    sdk = ModelSdk()
    register_kokoro(sdk.registry)
    model_id = "hexgrad/kokoro"

    await sdk.resources.download_source(
        model_id,
        variant="v1.0",
        options={"runtime": "coreml"},
    )
    async with await sdk.load(model_id, runtime="coreml") as handle:
        synthesizer = handle.require(SpeechSynthesis)
        result = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text="Hello from Kokoro.",
                voice="af_heart",
            )
        )
        Path("kokoro-output.f32le").write_bytes(result.audio)


asyncio.run(main())
```

## Notes and limits

`af_heart` is the default voice. Speed control depends on the selected runtime;
unsupported settings return a normal SDK inference error.

## License

The model is licensed under `Apache-2.0`.
