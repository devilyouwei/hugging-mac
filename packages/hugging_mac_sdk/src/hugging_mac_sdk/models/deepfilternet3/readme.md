# DeepFilterNet3

## Summary

Canonical package for `deepfilternet/deepfilternet3`, containing the prebuilt Core ML graph, complete libdf-compatible DSP pipeline, and managed-resource lifecycle.

## Runtimes and variants

- Variant: `default`
- Runtime: `coreml`
- Artifact: prebuilt INT8 Core ML model

## Capability API

The instance exposes the public `SpeechEnhancement` capability through `enhance_speech(...)`. It accepts encoded audio at any supported input sample rate, processes mono audio at the model's native 48 kHz, and returns PCM WAV at the requested rate (the source rate by default).

The inference path is:

1. Decode, downmix, and resample to 48 kHz.
2. Run the 960-point STFT and build normalized ERB/spectral features.
3. Predict the ERB mask and five-tap deep-filter coefficients with Core ML.
4. Apply the mask and complex deep filter, then reconstruct with iSTFT.
5. Compensate algorithmic delay, preserve input duration, and encode PCM WAV.

## Package structure

- `__init__.py`: definition, manifest, and registration exports.
- `model.yaml`: identity and prebuilt artifact metadata.
- `config.py`: Core ML instance/resource options.
- `definition.py`: Core ML factory and registry binding.
- `instance.py`: lifecycle plus the typed speech-enhancement capability.
- `coreml.py`: thread-affine compiled graph loading and inference.
- `dsp.py`: STFT, feature normalization, ERB mask, deep filtering, and iSTFT.
- `audio.py`: audio decoding, resampling, and WAV encoding.
- `resources.py`: artifact validation, status, download, and deletion.
