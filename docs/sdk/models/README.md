# Model SDK Reference

Every example follows the same public workflow: register a model, prepare its
files, load a handle, require a capability, and submit a typed request.

## Audio and speech

| Model | Capability | Guide |
|---|---|---|
| Audio8 ASR | Speech transcription | [Audio8 ASR](audio8-asr.md) |
| Audio8 TTS | Speech synthesis | [Audio8 TTS](audio8-tts.md) |
| DeepFilterNet3 | Speech enhancement | [DeepFilterNet3](deepfilternet3.md) |
| Kokoro | Speech synthesis | [Kokoro](kokoro.md) |
| Nemotron 3.5 ASR | Streaming transcription | [Nemotron 3.5 ASR](nemotron-3.5-asr.md) |
| Qwen3 ASR | Speech transcription | [Qwen3 ASR](qwen3-asr.md) |
| Qwen3 TTS | Speech synthesis | [Qwen3 TTS](qwen3-tts.md) |
| SenseVoice | Transcription and speech understanding | [SenseVoice](sensevoice.md) |
| Silero VAD | Voice activity detection | [Silero VAD](silero-vad.md) |

## Language and multimodal

| Model | Capability | Guide |
|---|---|---|
| Gemma 4 | Chat | [Gemma 4](gemma-4.md) |
| Qwen3.5 | Chat and image understanding | [Qwen3.5](qwen3.5.md) |

## Vision

| Model | Capability | Guide |
|---|---|---|
| MediaPipe Hand Detection | Hand detection | [MediaPipe Hand Detection](mediapipe-hand-detection.md) |
| RetinaFace | Face detection | [RetinaFace](retinaface.md) |
| YOLOv8 | Object detection | [YOLOv8](yolov8.md) |
| YOLOv8 Pose | Pose estimation | [YOLOv8 Pose](yolov8-pose.md) |
| YOLOv8 Seg | Instance segmentation | [YOLOv8 Seg](yolov8-seg.md) |

Common lifecycle, resource, and error handling is covered in the
[SDK guide](../README.md).
