import time
from collections import deque

import lightning.pytorch as pl


class StepTimer(pl.Callback):
    """Logs per-step wall-clock timing to surface compute vs. data-loading bottlenecks.

    Measures two intervals every training batch:
    - ``perf/step_time_s``: time from batch_start to batch_end (forward + backward + optimizer).
    - ``perf/data_time_s``: time from previous batch_end to batch_start (data loading + collation).

    A rolling mean over the last ``window`` steps is logged every ``log_every_n_steps`` steps
    on rank 0 only.

    Args:
        log_every_n_steps: How often (in optimizer steps) to emit the rolling averages.
        window: Rolling window size for the averages.
        sync_cuda: If True, call ``torch.cuda.synchronize()`` before recording timestamps
            for accurate GPU-inclusive timing. Adds a small overhead; useful for profiling.
    """

    def __init__(self, log_every_n_steps: int = 50, window: int = 50, sync_cuda: bool = False):
        self._log_every = log_every_n_steps
        self._sync_cuda = sync_cuda
        self._step_times: deque[float] = deque(maxlen=window)
        self._data_times: deque[float] = deque(maxlen=window)
        self._batch_start: float | None = None
        self._batch_end: float | None = None

    def _now(self) -> float:
        if self._sync_cuda:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = self._now()
        if self._batch_end is not None:
            self._data_times.append(now - self._batch_end)
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_end = self._now()
        if self._batch_start is not None:
            self._step_times.append(self._batch_end - self._batch_start)

        if (batch_idx + 1) % self._log_every != 0:
            return
        if not self._step_times:
            return

        avg_step = sum(self._step_times) / len(self._step_times)
        pl_module.log("perf/step_time_s", avg_step, on_step=True, on_epoch=False, rank_zero_only=True)

        if self._data_times:
            avg_data = sum(self._data_times) / len(self._data_times)
            pl_module.log("perf/data_time_s", avg_data, on_step=True, on_epoch=False, rank_zero_only=True)
            pl_module.log(
                "perf/data_frac",
                avg_data / (avg_step + avg_data),
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset so the first batch of a new epoch doesn't count inter-epoch idle time as data time.
        self._batch_end = None
