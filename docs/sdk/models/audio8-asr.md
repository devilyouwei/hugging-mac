# Audio8 ASR

## Purpose

Audio8 ASR is a lightweight multilingual model for short speech transcription.

## Support

- Model ID: `audio8/audio8-asr`
- Variants: `0.1b` (default)
- Runtimes: `pytorch-mps`, `coreml` (default)
- Capabilities: `speech-transcription` through `SpeechTranscription`
- Platform: macOS; Core ML requires Apple Silicon
- Optional dependency set: `asr`

## Prepare resources

The PyTorch source can be downloaded directly. Core ML can be downloaded when a
prebuilt package is available or created through the registered converter.

## Example

```python
import asyncio

from hugging_mac_sdk import AudioInput, ModelSdk, TranscriptionRequest
from hugging_mac_sdk.capabilities import SpeechTranscription
from hugging_mac_sdk.models.audio8_asr import register_audio8_asr


async def main() -> None:
    sdk = ModelSdk()
    register_audio8_asr(sdk.registry)
    model_id = "audio8/audio8-asr"

    await sdk.resources.download_source(model_id, variant="0.1b")
    async with await sdk.load(
        model_id,
        variant="0.1b",
        runtime="pytorch-mps",
    ) as handle:
        transcriber = handle.require(SpeechTranscription)
        result = await transcriber.transcribe(
            TranscriptionRequest(audio=AudioInput(path="speech.wav"))
        )
        print(result.text)


asyncio.run(main())
```

## Notes and limits

The model is intended for audio clips up to 30 seconds. It returns transcript
text rather than word-level timestamps.

## License

The model is licensed under `CC-BY-NC-4.0`. Review the upstream terms before
commercial use or redistribution.
