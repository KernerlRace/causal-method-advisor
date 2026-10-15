"""Потолок нагрузки: во время обучения компьютером пользуются и для другого.

GPU: после каждого микробатча — пауза, чтобы видеокарта считала не больше доли duty времени
(0.75 → около 75% в nvidia-smi), плюс потолок памяти: лишнее — ошибка, а не уход в общую память Windows.
CPU: потоков PyTorch не больше доли логических ядер.
"""
from __future__ import annotations

import os
import time

import torch


def limit_cpu(fraction: float = 0.75) -> int:
    n = max(1, int((os.cpu_count() or 1) * fraction))
    n = min(n, torch.get_num_threads())
    torch.set_num_threads(n)
    return n


class GpuDuty:
    """Вызывать step() после каждого микробатча: спит busy·(1−duty)/duty, где busy — время с прошлого вызова.
    Паузы короткие и частые (десятки мс), поэтому средняя загрузка в nvidia-smi ровная, без пиков."""

    def __init__(self, device: torch.device, duty: float = 0.65, memory_fraction: float = 0.75):
        self.enabled = device.type == "cuda" and duty < 1.0
        self.duty = duty
        if device.type == "cuda" and memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device.index or 0)
        self.t = time.perf_counter()

    def step(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        busy = time.perf_counter() - self.t
        time.sleep(busy * (1.0 - self.duty) / self.duty)
        self.t = time.perf_counter()
