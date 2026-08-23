# RetinaFace Technical Notes

The package implements `FaceDetection` with PyTorch and Core ML engines composed
under one instance. Static facts are owned by `model.yaml`.

`utils/modeling.py` contains the local MobileNet-based network used for trusted
checkpoint loading and conversion. `utils/preprocess.py` owns fixed-size
letterboxing and normalization. `utils/postprocess.py` decodes priors, filters
scores, applies NMS, and restores boxes and five alignment landmarks to source
pixels.

`converter.py` exports the model-specific raw-output boundary. Both runtime
engines return equivalent private values to `instance.py`, which constructs the
public face response. `resources.py` manages injected source/target artifacts and
never repeats source or file constants.
