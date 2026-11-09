"""Sudo-free power sampling on Apple Silicon via `macmon pipe` (github.com/vladkens/macmon).

macmon reads the same IOReport energy counters that `powermetrics` uses, without
root. Each sample is the average over its interval, so energy over a window is
the sum of power x interval for the samples whose timestamps fall inside it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

FIELDS = ("gpu_power", "cpu_power", "ram_power", "all_power", "sys_power", "gpu_freq_mhz", "gpu_active_ratio")


@dataclass
class Sample:
    t: float  # time.monotonic() when the sample arrived (end of its interval)
    gpu_power: float
    cpu_power: float
    ram_power: float
    all_power: float  # CPU + GPU + ANE + DRAM (SoC package)
    sys_power: float  # whole machine as reported by the SMC
    gpu_freq_mhz: float
    gpu_active_ratio: float


class PowerSampler:
    """Background thread collecting macmon samples; use as a context manager."""

    def __init__(self, interval_ms: int = 100):
        exe = shutil.which("macmon")
        if exe is None:
            raise RuntimeError("macmon not found: brew install macmon")
        self.interval_s = interval_ms / 1000
        self._cmd = [exe, "pipe", "-i", str(interval_ms)]
        self.samples: list[Sample] = []
        self._lock = threading.Lock()

    def __enter__(self) -> "PowerSampler":
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(4 * self.interval_s)  # let the first samples arrive
        return self

    def __exit__(self, *exc) -> None:
        self._proc.terminate()
        self._proc.wait(timeout=5)

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = Sample(time.monotonic(), *(float(d.get(k, 0.0)) for k in FIELDS))
            with self._lock:
                self.samples.append(s)

    def window(self, t0: float, t1: float) -> list[Sample]:
        """Samples whose interval lies mostly inside [t0, t1]."""
        half = self.interval_s / 2
        with self._lock:
            return [s for s in self.samples if t0 + half <= s.t <= t1 + half]

    def mean(self, t0: float, t1: float) -> dict[str, float]:
        w = self.window(t0, t1)
        if not w:
            raise RuntimeError("no power samples in window; run longer or shorten the interval")
        return {k: sum(getattr(s, k) for s in w) / len(w) for k in FIELDS} | {"n_samples": len(w)}

    def wait_for_next(self, n: int = 2) -> None:
        """Block until n fresh samples have arrived (aligns windows to sample edges)."""
        with self._lock:
            start = len(self.samples)
        while True:
            with self._lock:
                if len(self.samples) >= start + n:
                    return
            time.sleep(self.interval_s / 4)
