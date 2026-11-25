---
license: apache-2.0
language:
- zh
- en
base_model: MigoXV/icefall-kws-zipformer-zh-en-3M-2025-12-20
pipeline_tag: audio-classification
tags:
- coreml
- keyword-spotting
- wake-word
- zipformer
- apple-silicon
---

# Zipformer KWS zh-en 3M Core ML

Streaming keyword spotting for configured Mandarin and English phrases. The
bundle contains three native Core ML ML Programs (encoder, decoder, joiner)
ported from the original Zipformer weights with FP32 execution and batch size
one. The package deployment targets are iOS 17 and macOS 14. Validation uses
Apple Silicon macOS.

## Try the streaming spotter on a Mac

Clone or download the entire bundle. The Python example needs a 16 kHz
mono 16-bit PCM WAV. It uses the Core ML graphs for inference and a small
Python streaming host for filterbanks and contextual beam search.

```sh
uv run --no-project --with coremltools==9.0 --with kaldi-native-fbank==1.23.3 \
  python spotter.py sample.wav
```

Use `--keywords your-keywords.txt` to configure more phrases. Each line
contains space-separated tokens from `tokens.txt` and can end with `@label`.
English uses phones and Mandarin uses split pinyin tokens. Optional
`:boost` and `#threshold` settings apply to each phrase. The example emits
keyword matches and token timestamps as JSON lines. The CLI pads the tail
with silence, matching the bundled smoke run. Live clients can supply
short packets with `Stream.accept()` and keep a separate stream per session.

## Core ML graph interface

`manifest.json` records exact tensor names, input shapes and dtypes, plus
ordered outputs. Supply 29 frames of 80-bin Kaldi filterbanks and advance
16 frames for each encoder call. Initial caches are zero, then all
38 cache outputs are fed back in order. Use Int32 for token IDs and integer
caches. The decoder consumes the last two hypothesis tokens. The joiner
takes encoder and decoder projections and returns unnormalized vocabulary
logits. Timing, streaming state, and matching are host responsibilities.

## Validation

`evaluation.json` measures FP32 graph parity against the official Sherpa-ONNX
chunk-8 graphs and checks the detected labels and token timestamps on the
official audio fixtures and silence. It reports per-graph predict-call
P95 after warmup. Keyword quality and audio-to-detection delay depend on
the phrases, audio, and streaming configuration. This bundle provides
configured phrase detection for applications that implement their own
wake or interrupt policy.

See `NOTICE` and `manifest.json` for licensing and provenance.
