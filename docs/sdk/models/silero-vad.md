# Silero VAD

## Purpose

Silero VAD identifies speech intervals in recorded or streaming audio.

## Support

- Model ID: `snakers4/silero-vad`
- Variants: `v6.2.1` (default)
- Runtimes: `coreml` (default), `onnx`
- Capabilities: `voice-activity-detection` through `VoiceActivityDetection`
- Platform: Core ML requires Apple Silicon; ONNX also supports CPU execution
- Optional dependency set: `vad`

## Prepare resources

Select the runtime while downloading the model files.

## Example

```python
import asyncio

from hugging_mac_sdk import AudioInput, ModelSdk, VoiceActivityRequest
from hugging_mac_sdk.capabilities import VoiceActivityDetection
from hugging_mac_sdk.models.silero import register_silero


async def main() -> None:
    sdk = ModelSdk()
    register_silero(sdk.registry)
    model_id = "snakers4/silero-vad"

    await sdk.resources.download_source(
        model_id,
        variant="v6.2.1",
        options={"runtime": "coreml"},
    )
    async with await sdk.load(model_id, runtime="coreml") as handle:
        vad = handle.require(VoiceActivityDetection)
        result = await vad.detect_voice_activity(
            VoiceActivityRequest(
                audio=AudioInput(path="meeting.wav"),
                threshold=0.5,
                min_speech_ms=250,
                min_silence_ms=100,
            )
        )
        print([(item.start_seconds, item.end_seconds) for item in result.segments])


asyncio.run(main())
```

## Notes and limits

Tune thresholds for the recording environment. For continuous audio, preserve
chunk order and reset application state between unrelated streams.

## License

The model is licensed under `MIT`.
