"""NumPy DSP frontend and backend matching DeepFilterNet/libdf conventions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

FFT_SIZE = 960
HOP_SIZE = 480
FREQUENCY_BINS = 481
ERB_BANDS = 32
DF_BINS = 96
DF_ORDER = 5
DF_LOOKAHEAD = 2
NATIVE_SAMPLE_RATE = 48000
MAX_FRAMES = 6000
NORM_ALPHA = np.float32(0.99)

FloatArray = NDArray[np.float32]
Predict = Callable[[dict[str, NDArray[np.float16]]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class EnhancementOutput:
    samples: FloatArray
    neural_inference_ms: float


class DeepFilterNet3Dsp:
    """Complete STFT → Core ML features → mask/DF → iSTFT pipeline."""

    def __init__(self, auxiliary_path: Path) -> None:
        with np.load(auxiliary_path) as auxiliary:
            self._erb_fb = self._read_array(auxiliary, "erb_fb", (FREQUENCY_BINS, ERB_BANDS))
            self._erb_inv_fb = self._read_array(
                auxiliary, "erb_inv_fb", (ERB_BANDS, FREQUENCY_BINS)
            )
            self._window = self._read_array(auxiliary, "window", (FFT_SIZE,))
            self._mean_norm_init = self._read_array(auxiliary, "mean_norm_state", (ERB_BANDS,))
            unit = np.asarray(auxiliary["unit_norm_state"], dtype=np.float32).reshape(-1)
            if unit.shape != (DF_BINS,):
                raise ValueError(f"unit_norm_state has shape {unit.shape}; expected {(DF_BINS,)}")
            self._unit_norm_init = np.ascontiguousarray(unit)

    @staticmethod
    def _read_array(archive: Any, name: str, shape: tuple[int, ...]) -> FloatArray:
        value = np.asarray(archive[name], dtype=np.float32)
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
        return np.ascontiguousarray(value)

    def enhance(self, samples: FloatArray, predict: Predict) -> EnhancementOutput:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("DeepFilterNet3 requires non-empty mono audio")
        if not bool(np.isfinite(samples).all()):
            raise ValueError("DeepFilterNet3 input contains non-finite samples")

        spectrum = self._stft(samples)
        frames = spectrum.shape[0]
        if frames > MAX_FRAMES:
            raise ValueError(
                f"DeepFilterNet3 input produces {frames} frames; Core ML maximum is {MAX_FRAMES}"
            )

        erb = self._erb_features(spectrum)
        spec = self._spectral_features(spectrum[:, :DF_BINS])
        inputs = {
            "feat_erb": np.ascontiguousarray(erb[None, None], dtype=np.float16),
            "feat_spec": np.ascontiguousarray(
                np.stack((spec.real, spec.imag), axis=0)[None], dtype=np.float16
            ),
        }
        started = perf_counter()
        raw = predict(inputs)
        neural_ms = (perf_counter() - started) * 1000
        try:
            mask = np.asarray(raw["erb_mask"], dtype=np.float32).reshape(frames, ERB_BANDS)
            coefficients = np.asarray(raw["df_coefs"], dtype=np.float32).reshape(
                DF_ORDER, frames, DF_BINS, 2
            )
        except (KeyError, ValueError) as error:
            shapes = {name: getattr(value, "shape", None) for name, value in raw.items()}
            raise ValueError(f"Unexpected DeepFilterNet3 outputs: {shapes}") from error

        enhanced = spectrum * (mask @ self._erb_inv_fb).astype(np.float32)
        enhanced[:, :DF_BINS] = self._apply_deep_filter(spectrum, coefficients)
        output = self._istft(enhanced)
        # Remove the streaming STFT delay and retain the exact input length.
        output = output[FFT_SIZE - HOP_SIZE : FFT_SIZE - HOP_SIZE + samples.size]
        if output.size != samples.size:
            raise ValueError(
                f"DeepFilterNet3 output length {output.size} does not match input {samples.size}"
            )
        output = np.nan_to_num(output, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return EnhancementOutput(np.ascontiguousarray(output, dtype=np.float32), neural_ms)

    def _stft(self, samples: FloatArray) -> NDArray[np.complex64]:
        # libdf begins with one overlap buffer of zeros and flushes with a full FFT window.
        buffer = np.concatenate(
            (
                np.zeros(FFT_SIZE - HOP_SIZE, dtype=np.float32),
                samples,
                np.zeros(FFT_SIZE, dtype=np.float32),
            )
        )
        frames = np.lib.stride_tricks.sliding_window_view(buffer, FFT_SIZE)[::HOP_SIZE]
        windowed = np.asarray(frames * self._window, dtype=np.float32)
        return np.asarray(np.fft.rfft(windowed, n=FFT_SIZE, axis=1) / FFT_SIZE, dtype=np.complex64)

    def _istft(self, spectrum: NDArray[np.complex64]) -> FloatArray:
        # norm="forward" leaves inverse DFT unscaled, matching libdf/vDSP synthesis.
        frames = np.fft.irfft(spectrum, n=FFT_SIZE, axis=1, norm="forward").astype(
            np.float32, copy=False
        )
        frames *= self._window
        output = np.empty(frames.shape[0] * HOP_SIZE, dtype=np.float32)
        memory = np.zeros(FFT_SIZE - HOP_SIZE, dtype=np.float32)
        for index, frame in enumerate(frames):
            frame = frame.copy()
            frame[: memory.size] += memory
            output[index * HOP_SIZE : (index + 1) * HOP_SIZE] = frame[:HOP_SIZE]
            memory = frame[HOP_SIZE:]
        return output

    def _erb_features(self, spectrum: NDArray[np.complex64]) -> FloatArray:
        power = np.square(spectrum.real) + np.square(spectrum.imag)
        erb = 10.0 * np.log10(power @ self._erb_fb + np.float32(1e-10))
        state = self._mean_norm_init.copy()
        one_minus_alpha = np.float32(1.0) - NORM_ALPHA
        for frame in erb:
            state *= NORM_ALPHA
            state += frame * one_minus_alpha
            frame -= state
            frame /= np.float32(40.0)
        return np.ascontiguousarray(erb, dtype=np.float32)

    def _spectral_features(self, spectrum: NDArray[np.complex64]) -> NDArray[np.complex64]:
        features = spectrum.copy()
        state = self._unit_norm_init.copy()
        one_minus_alpha = np.float32(1.0) - NORM_ALPHA
        for frame in features:
            state *= NORM_ALPHA
            state += np.abs(frame) * one_minus_alpha
            frame /= np.sqrt(np.maximum(state, np.float32(1e-10)))
        return np.ascontiguousarray(features, dtype=np.complex64)

    @staticmethod
    def _apply_deep_filter(
        spectrum: NDArray[np.complex64], coefficients: FloatArray
    ) -> NDArray[np.complex64]:
        frames = spectrum.shape[0]
        result = np.zeros((frames, DF_BINS), dtype=np.complex64)
        source_bins = spectrum[:, :DF_BINS]
        for tap in range(DF_ORDER):
            shift = tap - (DF_ORDER - 1 - DF_LOOKAHEAD)
            shifted = np.zeros_like(source_bins)
            if shift < 0:
                shifted[-shift:] = source_bins[: frames + shift]
            elif shift > 0:
                shifted[: frames - shift] = source_bins[shift:]
            else:
                shifted[:] = source_bins
            weight = coefficients[tap, :, :, 0] + 1j * coefficients[tap, :, :, 1]
            result += shifted * weight
        return result
