# Zipformer conversion

This package owns the fixed batch-one bilingual streaming KWS export and its
Hub publication gate. The encoder takes a rolling window of 80-bin features
and 38 caches. The prediction network takes two token IDs. The joiner returns
vocabulary logits for one encoder frame and one decoder context. Keep all
three graphs together. The host feeds back the encoder cache outputs in IO
order and owns audio features, keyword definitions, and beam search.

`model.yaml` pins the source code and weights, reference archive, and target.
`converter.py` exports the three graphs. `release.py` stages the bundle,
checks parity, and owns the explicit publish command. The existing generic
conversion CLI serves single-output decision graphs. Zipformer uses a
model-specific entry point as a three-graph bundle and adds no SDK registration
or dependency on the Decision models.

The release targets native Core ML ML Programs with FP32 CPU execution. Host
keyword matching is separate from application intent policy. Source archives,
export runs, and staged checks follow the project's ignored `tmp/` layout.

## Build and validate

Run from the Hugging Mac repository root. The recipe declares the temporary
ONNX runtime and filterbank tool dependencies at the invocation. The other
tool versions come from the conversion workbench lockfile.

```sh
uv run --project packages/hugging_mac_conversion --locked \
  --with onnxruntime==1.24.4 --with kaldi-native-fbank==1.23.3 \
  python -m hugging_mac_conversion.models.zipformer.release build
```

Once both pinned sources are present under `tmp/originals`, pass `--offline`
before `build` to rebuild without downloading. Each run writes a new directory
under `tmp/conversions` and prints its path. `build.json` records the validation
state and the file inventory. `manifest.json` records provenance, tool versions
and the ordered graph interface. `evaluation.json` records the native and
reference comparisons.

To independently verify the bundle, use the same invocation with
`--offline verify <run>` in place of `build`. The check compares all 39 encoder
outputs, decoder projections, and joiner logits with the official FP32 ONNX
graphs. It checks Mandarin and English detections, reported token timestamps,
and negatives on the official samples and silence. Graph P95 measures warm
predict calls on the validation Mac. Audio-to-detection latency and application
intent quality require their own evaluation.

## Publish

Use `--offline push <run>` in the invocation above to rerun validation and
show the Hub destination. After authenticating with a Hugging Mac write
account, append `--execute` to upload all three packages and their assets in
one commit. The release card and Python spotter ship with the bundle.
