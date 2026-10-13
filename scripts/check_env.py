"""Проверка окружения (этап 0): версии библиотек, CUDA, быстрый тест обучения на GPU.

Запуск: .venv/Scripts/python.exe scripts/check_env.py
"""
import importlib
import platform
import sys
import time

REQUIRED = ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "cvxpy",
            "yaml", "streamlit", "matplotlib", "tqdm", "reportlab", "pytest"]
OPTIONAL = ["dowhy", "econml"]  # нужны только на бонус-этапе (расчёт эффекта)


def check_imports() -> bool:
    ok = True
    print(f"Python {platform.python_version()} ({sys.executable})")
    for name in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok] {name:12s} {getattr(mod, '__version__', '?')}")
        except ImportError as e:
            mark = "--" if name in OPTIONAL else "FAIL"
            print(f"  [{mark}] {name:12s} {e}")
            ok &= name in OPTIONAL
    return ok


def check_torch() -> bool:
    import torch

    print(f"\ntorch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA недоступна: стоит CPU-сборка torch или проблема с драйвером")
        return False
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"  [ok] {props.name}, {props.total_memory / 2**30:.1f} GB, compute {props.major}.{props.minor}")

    # мини-обучение: маленький MLP в fp16 autocast, loss должен упасть
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device=dev)
    y = (x[:, :8].sum(dim=1, keepdim=True) > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.perf_counter()
    for step in range(300):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == 0:
            first = loss.item()
    torch.cuda.synchronize()
    print(f"  [ok] fp16 training: loss {first:.3f} -> {loss.item():.3f} за {time.perf_counter() - t0:.1f} с")

    # грубая оценка пропускной способности матричного умножения
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
    print(f"  [ok] fp16 matmul: ~{tflops:.1f} TFLOPS")
    return loss.item() < first


if __name__ == "__main__":
    ok = check_imports()
    ok &= check_torch()
    print("\nОКРУЖЕНИЕ ГОТОВО" if ok else "\nЕСТЬ ПРОБЛЕМЫ (см. FAIL выше)")
    sys.exit(0 if ok else 1)
