"""Этап 6: GPT-подобный декодер с нуля. Всё на CPU и на крошечных моделях — секунды."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.data import TokenStream, write_tokens  # noqa: E402
from model.train_pretrain import lr_at  # noqa: E402
from model.transformer import (PRESETS, GPT, CausalSelfAttention, GPTConfig, apply_rope,  # noqa: E402
                               make_optimizer, rope_tables)

TINY = dict(vocab_size=97, context=48, n_layer=2, n_head=4, d_model=32)


def tiny(**kw) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**TINY, **kw})).eval()


def tokens(*shape, vocab=TINY["vocab_size"], seed=0):
    return torch.randint(0, vocab, shape, generator=torch.Generator().manual_seed(seed))


def test_base_preset_is_about_30m_with_tied_embeddings():
    model = GPT(GPTConfig(**PRESETS["base"]))
    assert 25e6 < model.num_params() < 35e6
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.cfg.mlp_hidden % 64 == 0
    with pytest.raises(ValueError):
        GPTConfig(d_model=100, n_head=8)


def test_untrained_loss_is_close_to_uniform_guess():
    """Малые веса на старте → почти равномерное распределение → loss ≈ ln(vocab)."""
    model = tiny()
    x, y = tokens(4, 32), tokens(4, 32, seed=1)
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(TINY["vocab_size"])) < 0.1


def test_future_tokens_do_not_leak_into_the_past():
    model = tiny()
    x = tokens(1, 24)
    x2 = x.clone()
    x2[0, 15:] = (x2[0, 15:] + 1) % TINY["vocab_size"]
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-6)
    assert not torch.allclose(a[0, 15:], b[0, 15:], atol=1e-3)


def test_manual_attention_matches_flash_kernel():
    manual = tiny(attention="manual")
    fused = tiny(attention="sdpa")
    fused.load_state_dict(manual.state_dict())
    x = tokens(3, 40)
    assert torch.allclose(manual(x)[0], fused(x)[0], atol=1e-5)


def test_attention_rows_are_causal_distributions():
    q, k = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    w = CausalSelfAttention.attention_weights(q, k)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 7))
    assert (w.triu(diagonal=1) == 0).all()


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = rope_tables(64, 16, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert score(5, 2) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)
    # поворот не меняет длину вектора
    assert torch.allclose(apply_rope(q, cos[9:10], sin[9:10]).norm(), q.norm())


def test_ignored_targets_do_not_count():
    model = tiny()
    x, y = tokens(2, 16), tokens(2, 16, seed=1)
    logits, _ = model(x)
    masked = y.clone()
    masked[:, :10] = -100
    _, loss = model(x, masked)
    expected = torch.nn.functional.cross_entropy(logits[:, 10:].reshape(-1, logits.shape[-1]), y[:, 10:].reshape(-1))
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


@pytest.mark.parametrize("attention", ["manual", "sdpa"])
def test_kv_cache_gives_the_same_logits_as_a_full_pass(attention):
    from model.transformer import KVCache

    model = tiny(attention=attention)
    x = tokens(2, 30)
    full, _ = model(x)
    cache = KVCache(model.cfg.n_layer)
    parts = [model(x[:, :12], cache=cache)[0]]  # промпт целиком, дальше по токену и кусками
    for i in range(12, 20):
        parts.append(model(x[:, i:i + 1], cache=cache)[0])
    parts.append(model(x[:, 20:], cache=cache)[0])
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-5)


def test_generation_with_and_without_cache_is_identical():
    model = tiny()
    prompt = tokens(1, 5)
    a = model.generate(prompt, 30, temperature=0, use_cache=True)
    b = model.generate(prompt, 30, temperature=0, use_cache=False)
    assert torch.equal(a, b)
    assert a.shape[1] == 35 and torch.equal(a[:, :5], prompt)
    # сэмплирование воспроизводимо с тем же генератором, top_k оставляет только k кандидатов
    s1 = model.generate(prompt, 10, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(1))
    assert torch.equal(s1, model.generate(prompt, 10, temperature=0))


def test_generation_stops_at_eos_and_context():
    model = tiny()
    prompt = tokens(1, 5)
    greedy = model.generate(prompt, 10, temperature=0)
    first = greedy[0, 5].item()
    stopped = model.generate(prompt, 10, temperature=0, eos_id=first)
    assert stopped.shape[1] == 6
    assert model.generate(prompt, 500, temperature=0).shape[1] == TINY["context"]
    with pytest.raises(ValueError):
        model(tokens(1, TINY["context"] + 1))


def test_overfits_a_single_batch():
    """Главная проверка этапа: модель способна запомнить что угодно — loss падает почти до нуля."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY)).train()
    opt = make_optimizer(model, lr=3e-3, weight_decay=0.0)
    seq = tokens(4, 33)
    x, y = seq[:, :-1], seq[:, 1:]
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 4.0
    assert loss.item() < 0.05


def test_optimizer_decays_only_matrices():
    model = tiny()
    decay, no_decay = make_optimizer(model, lr=1e-3, weight_decay=0.1).param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])  # веса RMSNorm
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == model.num_params()


def test_checkpoint_roundtrip(tmp_path):
    model = tiny()
    model.save(tmp_path / "m.pt", step=7)
    loaded, extra = GPT.load(tmp_path / "m.pt")
    loaded.eval()
    x = tokens(2, 20)
    assert torch.equal(model(x)[0], loaded(x)[0])
    assert extra["step"] == 7 and loaded.cfg == model.cfg
    assert loaded.lm_head.weight is loaded.tok_emb.weight


def test_lr_schedule():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup=10, total=110)
    assert lr_at(0, **kw) == pytest.approx(1e-4)
    assert lr_at(9, **kw) == pytest.approx(1e-3)
    assert lr_at(60, **kw) == pytest.approx(5.5e-4)
    assert lr_at(110, **kw) == pytest.approx(1e-4)
    assert lr_at(500, **kw) == pytest.approx(1e-4)
    assert all(lr_at(s, **kw) >= lr_at(s + 1, **kw) for s in range(10, 120))


def test_token_stream_windows_are_shifted_by_one(tmp_path):
    from model.data import TOKENIZER
    from model.tokenizer import BPETokenizer

    if not TOKENIZER.exists():
        pytest.skip("run python -m model.train_tokenizer first")
    tok = BPETokenizer.load(TOKENIZER)
    texts = ["Did the campaign raise sales?", "<Q>Повлияла ли акция на выручку?<ANS>"] * 20
    n = write_tokens(texts, tmp_path / "t.bin")
    stream = TokenStream(tmp_path / "t.bin")
    assert len(stream) == n
    eos = tok.token_id("<EOS>")
    assert tok.decode(stream.data[:n // 20].tolist()).split("<EOS>")[:2] == texts[:2]
    assert int((np.asarray(stream.data) == eos).sum()) == len(texts)
    x, y = stream.batch(5, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (5, 16) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
