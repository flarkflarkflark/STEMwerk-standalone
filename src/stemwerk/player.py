from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import sounddevice as sd


@dataclass
class StemState:
    solo: bool = False
    mute: bool = False
    volume: float = 1.0


class Player:
    def __init__(self) -> None:
        self._stream: Optional[sd.OutputStream] = None
        self._data: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._channels: int = 2
        self._position_samples: int = 0
        self._lock = threading.RLock()
        self._stems: Dict[str, np.ndarray] = {}
        self._stem_states: Dict[str, StemState] = {}
        self._levels: Dict[str, float] = {}
        self._output_device: Optional[int] = None
        self.last_error: Optional[str] = None
        self.on_position_changed: Optional[Callable[[float], None]] = None

    def load_audio(self, data: np.ndarray, sample_rate: int) -> None:
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        self._data = data.astype(np.float32, copy=False)
        self._sample_rate = sample_rate
        self._channels = self._data.shape[1]
        self._position_samples = 0

    def set_stems(self, stems: Dict[str, np.ndarray]) -> None:
        self._stems = {}
        self._stem_states = {}
        for name, data in stems.items():
            if data.ndim == 1:
                data = np.stack([data, data], axis=1)
            self._stems[name] = data.astype(np.float32, copy=False)
        for name in stems.keys():
            self._stem_states.setdefault(name, StemState())

    def update_stem_state(
        self,
        stem_name: str,
        solo: bool,
        mute: bool,
        volume: float,
    ) -> None:
        self._stem_states[stem_name] = StemState(
            solo=solo,
            mute=mute,
            volume=volume,
        )

    def clear_stems(self) -> None:
        self._stems = {}
        self._stem_states = {}
        self._levels = {}

    def prune_stem_states(self, keep: List[str]) -> None:
        self._stem_states = {name: self._stem_states.get(name, StemState()) for name in keep}

    def set_output_device(self, device_id: Optional[int]) -> None:
        self._output_device = device_id

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def position(self) -> float:
        return self._position_samples / float(self._sample_rate)

    def get_levels(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._levels)

    def get_stem_levels(self) -> Dict[str, float]:
        return self.get_levels()

    def set_position(self, seconds: float) -> None:
        with self._lock:
            self._position_samples = int(max(0.0, seconds) * self._sample_rate)

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._position_samples = int(max(0.0, seconds) * self._sample_rate)
        if self.is_playing:
            self.stop()
            self.play()

    def play(self) -> bool:
        if self._data is None:
            self.last_error = "No audio loaded."
            return False
        if self.is_playing:
            return True
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._output_device,
                callback=self._callback,
            )
            self._stream.start()
            self.last_error = None
            return True
        except Exception as exc:
            self._stream = None
            self.last_error = str(exc)
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _active_stems(self) -> Dict[str, StemState]:
        soloed = [name for name, state in self._stem_states.items() if state.solo and not state.mute]
        if soloed:
            return {name: self._stem_states[name] for name in soloed}
        return {name: state for name, state in self._stem_states.items() if not state.mute}

    def _get_mix_segment(self, start: int, frames: int) -> np.ndarray:
        if self._data is None:
            return np.zeros((frames, self._channels), dtype=np.float32)

        if not self._stems:
            end = start + frames
            segment = self._data[start:end]
            if segment.shape[0] < frames:
                pad = np.zeros((frames - segment.shape[0], self._channels), dtype=np.float32)
                segment = np.vstack([segment, pad])
            self._levels = {}
            return segment

        active = self._active_stems()
        if not active:
            self._levels = {name: 0.0 for name in self._stem_states}
            return np.zeros((frames, self._channels), dtype=np.float32)

        mix = np.zeros((frames, self._channels), dtype=np.float32)
        levels: Dict[str, float] = {name: 0.0 for name in self._stem_states}
        for name, state in active.items():
            data = self._stems.get(name)
            if data is None:
                continue
            end = start + frames
            segment = data[start:end]
            if segment.shape[0] < frames:
                pad = np.zeros((frames - segment.shape[0], self._channels), dtype=np.float32)
                segment = np.vstack([segment, pad])
            volume = float(state.volume)
            mix += segment * volume
            if segment.size > 0:
                rms = float(np.sqrt(np.mean(np.square(segment))))
                levels[name] = min(1.0, rms * volume)
        self._levels = levels
        return mix

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        if self._data is None:
            outdata[:] = np.zeros((frames, self._channels), dtype=np.float32)
            return
        with self._lock:
            start = self._position_samples
            segment = self._get_mix_segment(start, frames)
            outdata[:] = segment
            self._position_samples += frames
            if self._position_samples >= self._data.shape[0]:
                raise sd.CallbackStop()
