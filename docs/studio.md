# Studio Guide

The Studio combines local models into task-focused Apps and camera-driven Games.
Availability depends on which models and runtimes are prepared in the Model
Library.

## Apps

- **Object Detection** finds objects in images, video, or a live camera feed.
- **Pose Estimation** combines body, face, and hand landmarks for local motion
  analysis.
- **Instance Segmentation** returns object masks and contours.
- **Live Transcription** can combine speech enhancement, voice activity
  detection, and speech recognition.
- **Text to Speech** synthesizes and compares voices from supported TTS models.
- **Local Chat** supports private text and vision-language conversations, with
  optional speech input and spoken responses.

Each App reports when its required model files are missing. Prepare the requested
runtime in the Model Library, load an instance, and return to the App.

## Games

The Games section contains local, camera-driven experiences built from model
capabilities. Grant camera permission when prompted. The browser camera API works
directly on localhost; access from another device normally requires HTTPS.

## Images, audio, video, and camera input

Uploaded media is handled by the local backend. Camera and video workflows may
drop frames to remain responsive instead of building an ever-growing queue.
Closing a workflow or changing pages stops active capture and releases browser
media tracks.

Large or unsupported files are rejected before inference. Browser playback and
camera support also depend on the formats supported by the installed browser.

## Privacy and local data

- Inference runs on the Mac.
- Model downloads contact the source shown for that model.
- User media and generated files are not sent to an external inference service
  by default.
- Local caches can contain uploaded or generated media; remove them according to
  your own privacy and retention requirements.
- Prompts and media are not included in telemetry because external telemetry is
  disabled by default.

For model file operations, see the [Model Library guide](model-library.md).
