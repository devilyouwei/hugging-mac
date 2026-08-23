# Audio8 TTS Preview

## Purpose

Audio8 TTS generates multilingual speech and supports reference-audio voice
cloning on compatible runtimes.

## Support

- Model ID: `audio8/audio8-tts-preview`
- Variants: `0.1b-preview` (default), `0.6b-preview`
- Runtimes: `pytorch` (default), `mlx`
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Compatibility: both variants use CPU by default with PyTorch; `0.6b-preview` also supports MLX
- Platform: MLX requires Apple Silicon
- Optional dependency set: `tts`

## Prepare resources

Choose the same variant and runtime for resource preparation and loading. The
example uses the MLX build of the 0.6B variant.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import AudioInput, ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.models.audio8_tts import register_audio8_tts


async def main() -> None:
    sdk = ModelSdk()
    register_audio8_tts(sdk.registry)
    model_id = "audio8/audio8-tts-preview"

    await sdk.resources.download_source(
        model_id,
        variant="0.6b-preview",
        options={"runtime": "mlx"},
    )
    async with await sdk.load(
        model_id,
        variant="0.6b-preview",
        runtime="mlx",
    ) as handle:
        synthesizer = handle.require(SpeechSynthesis)
        result = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text="Hello from Audio8.",
                reference_audio=AudioInput(path="reference.wav"),
                reference_text="The exact transcript of the reference recording.",
            )
        )
        Path("audio8-output.f32le").write_bytes(result.audio)
        print(result.sample_rate, result.audio_format)


asyncio.run(main())
```

## Notes and limits

`reference_audio` and `reference_text` must be provided together. PyTorch uses
CPU by default because it currently performs better for both variants. Callers
can explicitly request MPS; after an MPS availability, loading, or inference
failure, the engine retries once on CPU. MLX GPU execution is limited to
`0.6b-preview`.

## License

The package manifest declares `Apache-2.0`. The 0.1B variant also carries
variant-specific upstream terms; review the model page before redistribution.
