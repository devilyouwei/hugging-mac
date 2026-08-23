# MOSS-TTS-Nano

## Purpose

MOSS-TTS-Nano generates multilingual speech directly or clones a voice when an
optional reference audio recording is supplied.

## Support

- Model ID: `openmoss-team/moss-tts-nano-100m`
- Variants: `nano-100m` (default)
- Runtimes: `mlx` (default)
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Platform: Apple Silicon macOS
- Optional dependency set: `tts`

## Prepare resources

Download the model and its audio tokenizer before loading. The SDK prepares the
shared tokenizer alongside the selected model files.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import AudioInput, ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.models.moss_tts_nano import register_moss_tts_nano


async def main() -> None:
    sdk = ModelSdk()
    register_moss_tts_nano(sdk.registry)
    model_id = "openmoss-team/moss-tts-nano-100m"

    await sdk.resources.download_source(model_id, variant="nano-100m")
    async with await sdk.load(model_id, runtime="mlx") as handle:
        synthesizer = handle.require(SpeechSynthesis)
        result = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text="Hello from MOSS-TTS-Nano.",
                reference_audio=AudioInput(path="reference.wav"),
            )
        )
        Path("moss-tts-nano.f32le").write_bytes(result.audio)


asyncio.run(main())
```

## Notes and limits

A reference audio recording is optional and its transcript is not used. With a
reference, generation uses voice-clone mode; without one, it uses direct
continuation generation. Generation runs on MLX while the audio tokenizer uses
its CPU execution path. Streaming is not currently exposed.

## License

The model is licensed under `Apache-2.0`.
