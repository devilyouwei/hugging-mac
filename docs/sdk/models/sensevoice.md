# SenseVoice Small

## Purpose

SenseVoice recognizes speech and can also report language, emotion, and acoustic
events.

## Support

- Model ID: `funaudiollm/sensevoice`
- Variants: `small` (default)
- Runtimes: `pytorch-mps` (default), `coreml`
- Capabilities: `speech-transcription`, `speech-understanding` through
  `SpeechTranscription` and `SpeechUnderstanding`
- Platform: macOS; Core ML requires Apple Silicon
- Optional dependency set: `asr`

## Prepare resources

Download the source for PyTorch MPS. Register a converter and prepare Core ML
explicitly when that runtime is preferred.

## Example

```python
import asyncio

from hugging_mac_sdk import AudioInput, ModelSdk, SpeechUnderstandingRequest
from hugging_mac_sdk.capabilities import SpeechUnderstanding
from hugging_mac_sdk.models.sensevoice import register_sensevoice


async def main() -> None:
    sdk = ModelSdk()
    register_sensevoice(sdk.registry)
    model_id = "funaudiollm/sensevoice"

    await sdk.resources.download_source(model_id, variant="small")
    async with await sdk.load(model_id, runtime="pytorch-mps") as handle:
        understanding = handle.require(SpeechUnderstanding)
        result = await understanding.understand_speech(
            SpeechUnderstandingRequest(
                audio=AudioInput(path="speech.wav"),
                language="auto",
            )
        )
        print(result.text, result.languages, result.emotion, result.events)


asyncio.run(main())
```

## Notes and limits

Use `SpeechTranscription` when only plain text is needed. Inputs are intended for
short clips; long recordings should be segmented by the application.

## License

The model uses `FunASR-Model-License-1.1`, which is separate from the Hugging Mac
repository license.
