"""Local wake-word detection for opt-in standby."""

from typing import Protocol
from hashlib import sha256
from pathlib import Path
from importlib import import_module
from collections.abc import Callable
from importlib.metadata import version

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import AudioArray, audio_to_float32


MODEL_DIRECTORY = (
    Path.home()
    / ".local/share/reachy-mini-conversation-app/wakeword/1.13.4-gigaspeech-standard"
    / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)
KEYWORD_TOKENS = "▁HE Y ▁RE A CH Y"
MODEL_HASHES = (
    "fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53",
    "1e721676515bcd42a186979733981213c66c80db680e1cc582dfedf3be76e678",
    "f61ebd3eed3773a44d088d53dfae92dbb6aec4839f4dcaee2d402414741663a3",
    "eae9da0c7e1e6c6a3f4cc42d167899c388f6c6701b94cb96320e4f55df79624c",
)


class _KeywordStream(Protocol):
    def accept_waveform(self, sample_rate: int, samples: NDArray[np.float32]) -> None: ...


class _KeywordSpotter(Protocol):
    def create_stream(self, keywords: str) -> _KeywordStream: ...

    def is_ready(self, stream: _KeywordStream) -> bool: ...

    def decode_stream(self, stream: _KeywordStream) -> None: ...

    def get_result(self, stream: _KeywordStream) -> str: ...


class WakeWordDetector:
    """Detect “Hey Reachy” locally with the reviewed Sherpa model."""

    def __init__(
        self,
        model_directory: Path = MODEL_DIRECTORY,
        *,
        spotter_factory: Callable[..., _KeywordSpotter] | None = None,
    ) -> None:
        """Load the fixed model contract and create an empty detector stream."""
        files = {
            "tokens": model_directory / "tokens.txt",
            "encoder": model_directory / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            "decoder": model_directory / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            "joiner": model_directory / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Wake-word model is incomplete: {', '.join(missing)}")
        for path, expected_hash in zip(files.values(), MODEL_HASHES, strict=True):
            if sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Wake-word model failed integrity validation: {path}")
        if spotter_factory is None:
            if version("sherpa-onnx") != "1.13.4":
                raise RuntimeError("Wake-word runtime must be sherpa-onnx 1.13.4")
            spotter_factory = import_module("sherpa_onnx").KeywordSpotter
        self._spotter = spotter_factory(
            **{name: str(path) for name, path in files.items()},
            keywords_file="",
            num_threads=1,
            sample_rate=16000,
            keywords_score=1.0,
            keywords_threshold=0.20,
            provider="cpu",
        )
        self.reset()

    def reset(self) -> None:
        """Discard prior audio and arm a fresh detector stream."""
        self._stream = self._spotter.create_stream(KEYWORD_TOKENS)

    def accept(self, sample_rate: int, frame: AudioArray) -> bool:
        """Consume one recorder frame and report a wake-word match."""
        samples = audio_to_float32(frame)
        if samples.ndim == 2:
            if samples.shape[1] > samples.shape[0]:
                samples = samples.T
            samples = samples.mean(axis=1, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("Wake-word audio must contain mono or interleaved channel samples")
        self._stream.accept_waveform(sample_rate, np.ascontiguousarray(samples, dtype=np.float32))
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                self.reset()
                return True
        return False
