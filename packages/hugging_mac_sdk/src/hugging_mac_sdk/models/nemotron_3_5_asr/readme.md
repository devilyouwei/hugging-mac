# Nemotron 3.5 ASR Technical Notes

The package implements both `SpeechTranscription` and
`StreamingSpeechTranscription` on one stateful Core ML engine. All tier and
resource declarations come from `model.yaml` and artifact metadata.

`coreml.py` loads preprocessing, encoder, and decoder/joint sessions; maintains
feature, encoder, and recurrent decoder caches; and performs greedy RNN-T
decoding. Tier-dependent tensor shapes are read from installed metadata rather
than copied into Python constants.

`instance.py` owns stream session IDs, per-session ordering, cancellation,
complete-audio adaptation, and mapping to public transcript events. Audio
decode/resample helpers and private engine output values remain in `utils/`.
`resources.py` validates and manages the selected injected bundle without
redeclaring its source.
