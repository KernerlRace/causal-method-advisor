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
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"  [ok] {props.name}, {props.total_memory / 2**30:.1f} GB, compute {props.major}.{props.minor}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")  # Apple Silicon (M1/M2/...)
        print("  [ok] Apple MPS")
    else:
        print("  [FAIL] нет ни CUDA, ни MPS: стоит CPU-сборка torch или проблема с драйвером")
        return False
    sync = torch.cuda.synchronize if dev.type == "cuda" else torch.mps.synchronize
    # на CUDA учим в fp16 со скейлером; на MPS (у M1 нет bf16) — в fp32
    amp_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    # мини-обучение: маленький MLP, loss должен упасть
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device=dev)
    y = (x[:, :8].sum(dim=1, keepdim=True) > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for step in range(300):
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == 0:
            first = loss.item()
    sync()
    print(f"  [ok] training test: loss {first:.3f} -> {loss.item():.3f} за {time.perf_counter() - t0:.1f} с")

    # грубая оценка пропускной способности матричного умножения
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    a @ a
    sync()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    sync()
    tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
    print(f"  [ok] fp16 matmul: ~{tflops:.1f} TFLOPS")

    # бенчмарк, близкий к нашей задаче: GPT-подобный стек (8 слоёв, d=512, контекст 512), forward+backward
    layer = torch.nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True, norm_first=True)
    gpt = torch.nn.TransformerEncoder(layer, 8, enable_nested_tensor=False).to(dev)
    gopt = torch.optim.AdamW(gpt.parameters(), lr=1e-4)
    batch, seq = 16, 512
    h = torch.randn(batch, seq, 512, device=dev)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(seq, device=dev)
    for i in range(13):  # 3 прогрева + 10 замеров
        if i == 3:
            sync()
            t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            out = gpt(h, mask=mask, is_causal=True)
        gopt.zero_grad(set_to_none=True)
        scaler.scale(out.float().pow(2).mean()).backward()
        scaler.step(gopt)
        scaler.update()
    sync()
    tok_s = 10 * batch * seq / (time.perf_counter() - t0)
    print(f"  [ok] transformer train step (~25M params): ~{tok_s / 1000:.0f}k tokens/s")
    return loss.item() < first


if __name__ == "__main__":
    ok = check_imports()
    ok &= check_torch()
    print("\nОКРУЖЕНИЕ ГОТОВО" if ok else "\nЕСТЬ ПРОБЛЕМЫ (см. FAIL выше)")
    sys.exit(0 if ok else 1)
