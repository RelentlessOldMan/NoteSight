"""audio_io.py -- load audio from disk (and the live-capture path).

load_audio(path) -> (mono float32, sr)
  - wav/ogg/flac (and anything libsndfile groks) via soundfile.
  - everything else (mp3/m4a/aac/...) is decoded by piping through the ffmpeg
    binary bundled with imageio-ffmpeg -- no system ffmpeg install needed.

record_loopback(sec) is kept for the live-research path (system audio capture).
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

# soundfile handles the common lossless/ogg containers directly.
try:
    import soundfile as sf
except Exception:  # pragma: no cover - dependency should be present
    sf = None

_SF_EXTS = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".w64"}


def _to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def _load_ffmpeg(path: str, sr: int) -> np.ndarray:
    """Decode any format ffmpeg understands to mono float32 at `sr`."""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-nostdin", "-v", "error",
        "-i", path,
        "-f", "f32le",      # raw 32-bit float PCM on stdout
        "-acodec", "pcm_f32le",
        "-ac", "1",          # downmix to mono
        "-ar", str(sr),      # resample
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}:\n{msg}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str, sr: int = 48000) -> tuple[np.ndarray, int]:
    """Load `path` as mono float32, resampled to `sr` where needed.

    Returns (mono, sr). Raises FileNotFoundError / RuntimeError on failure.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if sf is not None and ext in _SF_EXTS:
        data, file_sr = sf.read(path, dtype="float32", always_2d=False)
        mono = _to_mono(data)
        if file_sr != sr:
            mono = _resample(mono, file_sr, sr)
        return mono, sr

    # Fallback: let ffmpeg decode + resample straight to our target rate.
    return _load_ffmpeg(path, sr), sr


def _resample(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resample (good enough for onset detection)."""
    if src_sr == dst_sr or mono.size == 0:
        return mono
    n_dst = int(round(mono.size * dst_sr / src_sr))
    x_src = np.linspace(0.0, 1.0, mono.size, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    return np.interp(x_dst, x_src, mono).astype(np.float32)


def record_loopback(seconds: int, sr: int = 48000) -> tuple[np.ndarray, int]:
    """Capture `seconds` of system loopback audio (the live-research path)."""
    import soundcard as sc

    spk = sc.default_speaker()
    mic = sc.get_microphone(spk.name, include_loopback=True)
    with mic.recorder(samplerate=sr, channels=2) as rec:
        data = rec.record(numframes=sr * seconds)
    return _to_mono(data), sr
