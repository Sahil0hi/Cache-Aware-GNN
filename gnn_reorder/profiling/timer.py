"""
profiling/timer.py
Lightweight epoch timer that records wall-clock time per epoch and
optionally dumps a CSV summary to the results directory.
"""

import time
import os
import csv
from typing import List, Optional


class EpochTimer:
    """Records per-epoch wall-clock time and computes statistics."""

    def __init__(self, warmup: int = 10):
        """
        Args:
            warmup: Number of initial epochs to discard from statistics.
                    These epochs are still run but not included in mean/std.
        """
        self.warmup = warmup
        self._times: List[float] = []
        self._start: Optional[float] = None
        self._epoch = 0

    def start(self):
        """Call just before the forward+backward pass begins."""
        if self._epoch == 0:
            print(f"[EpochTimer] Warming up for {self.warmup} epochs...")
        self._start = time.perf_counter()

    def stop(self):
        """Call just after the optimizer step completes."""
        assert self._start is not None, "Call .start() before .stop()"
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._epoch += 1
        if self._epoch > self.warmup:
            self._times.append(elapsed_ms)
            if self._epoch % 10 == 0:
                print(
                    f"  Epoch {self._epoch:4d} | {elapsed_ms:7.2f} ms "
                    f"(mean so far: {self.mean_ms:.2f} ms)"
                )
        self._start = None

    @property
    def mean_ms(self) -> float:
        if not self._times:
            return float("nan")
        return sum(self._times) / len(self._times)

    @property
    def std_ms(self) -> float:
        if len(self._times) < 2:
            return float("nan")
        mean = self.mean_ms
        variance = sum((t - mean) ** 2 for t in self._times) / (len(self._times) - 1)
        return variance ** 0.5

    def summary(self) -> dict:
        return {
            "epochs_measured": len(self._times),
            "warmup_epochs": self.warmup,
            "mean_ms_per_epoch": round(self.mean_ms, 3),
            "std_ms_per_epoch": round(self.std_ms, 3),
            "min_ms": round(min(self._times), 3) if self._times else float("nan"),
            "max_ms": round(max(self._times), 3) if self._times else float("nan"),
        }

    def save_csv(self, path: str, extra_fields: Optional[dict] = None):
        """Append one summary row to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = self.summary()
        if extra_fields:
            row.update(extra_fields)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[EpochTimer] Saved timing row → {path}")
