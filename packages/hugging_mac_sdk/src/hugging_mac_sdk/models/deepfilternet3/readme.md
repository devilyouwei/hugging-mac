# DeepFilterNet3 Technical Notes

The package implements `SpeechEnhancement` with a Core ML inference engine and a
model-private DSP pipeline. Declarative facts are owned by `model.yaml`.

## Processing pipeline

1. `audio.py` decodes, downmixes, and resamples input audio.
2. `dsp.py` performs STFT and creates normalized ERB/spectral features.
3. `coreml.py` runs the compiled model through the Core ML provider.
4. `dsp.py` applies the ERB mask and deep-filter coefficients and reconstructs
   the waveform.
5. `audio.py` compensates delay, preserves duration, resamples when requested,
   and encodes the public WAV response.

`instance.py` owns lifecycle and serialization. `resources.py` validates and
manages only the injected prebuilt artifact. Intermediate arrays and DSP state do
not cross the capability boundary.
