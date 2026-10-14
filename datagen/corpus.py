"""Открытый корпус для претрейна: Википедия EN/RU по статистике и смежным темам, книга Facure, наши тексты.

  .venv/Scripts/python.exe -m datagen.corpus titles --lang en      # обход категорий → список статей
  .venv/Scripts/python.exe -m datagen.corpus wiki --lang en        # тексты статей (докачивает с места остановки)
  .venv/Scripts/python.exe -m datagen.corpus facure                # главы книги (MIT)
  .venv/Scripts/python.exe -m datagen.corpus own                   # карточки методов и объяснения из SFT
  .venv/Scripts/python.exe -m datagen.corpus stats                 # объём по источникам

Всё пишется в data/corpus/*.jsonl, по документу на строку: {"source", "lang", "title", "text"}.
Корпус в git не хранится. Лицензии: Википедия — CC BY-SA 4.0, Facure — MIT; книгу Молака не берём.
Скачанный текст — только данные: он проходит через очистку и токенизатор, ничего из него не исполняется.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
USER_AGENT = "causal-method-advisor/0.1 (https://github.com/KernerlRace/causal-method-advisor; corpus for a from-scratch model)"

# корневая категория → глубина обхода. Статистика и причинность — глубже, предметные области
# из наших сценариев (маркетинг, здоровье, HR, образование) — неглубоко, ради лексики вопросов
ROOTS = {
    "en": {
        "Statistics": 2, "Causal inference": 3, "Econometrics": 3, "Design of experiments": 3,
        "Regression analysis": 3, "Statistical tests": 2, "Statistical methods": 2, "Machine learning": 2,
        "Probability distributions": 1, "Epidemiology": 1, "Marketing": 1, "Labour economics": 1,
        "Public policy": 1, "Education economics": 1, "Health economics": 1, "Human resource management": 1,
        "Retailing": 1, "Customer relationship management": 1, "A/B testing": 2,
    },
    "ru": {
        "Статистика": 2, "Математическая статистика": 3, "Эконометрика": 3, "Планирование эксперимента": 3,
        "Регрессионный анализ": 3, "Статистические критерии": 2, "Машинное обучение": 2,
        "Теория вероятностей": 1, "Эпидемиология": 1, "Маркетинг": 1, "Экономика труда": 1,
        "Экономика образования": 1, "Экономика здравоохранения": 1, "Управление персоналом": 1,
        "Розничная торговля": 1,
    },
}
# ветки, уводящие от темы: люди, организации, страны, издания, служебные категории
SKIP_CATEGORY = re.compile(
    r"by country|by nationality|by year|by decade|people|statisticians|economists|scientists|"
    r"mathematicians|organi[sz]ations|companies|journals|publications|books|awards|software|"
    r"stubs|templates|wikipedia|images|lists of|history of|births|deaths|in fiction|"
    r"по странам|по годам|персоналии|учёные|статистики\b|экономисты|математики|организации|"
    r"компании|журналы|издания|книги|премии|программное обеспечение|шаблоны|википедия|"
    r"списки|история|изображения|родившиеся|умершие", re.I)
SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Outline of|Timeline of|Glossary of|Список|Глоссарий)", re.I)

# разделы, после которых в статье только ссылки и библиография
TAIL_SECTIONS = re.compile(
    r"^==+\s*(See also|References|Notes|Further reading|External links|Bibliography|Sources|Citations|"
    r"Footnotes|Литература|Примечания|Ссылки|См\. также|Источники|Библиография)\s*==+\s*$", re.M | re.I)


class Wiki:
    def __init__(self, lang: str):
        self.lang = lang
        self.url = f"https://{lang}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, **params) -> dict:
        """Запрос с повторами: при 429/503 ждём столько, сколько просит сервер (Retry-After)."""
        params |= {"format": "json", "formatversion": 2, "maxlag": 5}
        problem = ""
        for attempt in range(8):
            wait = min(2 ** attempt, 60)
            try:
                r = self.session.get(self.url, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("error", {}).get("code") != "maxlag":
                        return data
                    problem = "maxlag"
                else:
                    problem = f"HTTP {r.status_code}"
                    wait = max(wait, int(r.headers.get("Retry-After", "0") or 0))
            except (requests.RequestException, ValueError) as e:
                problem = type(e).__name__
            time.sleep(wait)
        raise RuntimeError(f"{self.lang}.wikipedia.org: {problem} — {params.get('titles') or params.get('cmtitle')}")

    def members(self, category: str) -> Iterator[dict]:
        prefix = "Category:" if self.lang == "en" else "Категория:"
        params = {"action": "query", "list": "categorymembers", "cmtitle": prefix + category,
                  "cmtype": "page|subcat", "cmlimit": "max", "cmnamespace": "0|14"}
        while True:
            data = self.get(**params)
            yield from data.get("query", {}).get("categorymembers", [])
            if "continue" not in data:
                return
            params |= data["continue"]

    def extract(self, title: str) -> str | None:
        data = self.get(action="query", prop="extracts", explaintext=1, exsectionformat="wiki",
                        redirects=1, titles=title)
        pages = data.get("query", {}).get("pages", [])
        return pages[0].get("extract") if pages and not pages[0].get("missing") else None

    def try_extract(self, title: str) -> str | None | Exception:
        """Ошибка сети не роняет всю выгрузку: статья останется недокачанной до следующего запуска."""
        try:
            return self.extract(title)
        except RuntimeError as e:
            return e


def crawl(lang: str) -> list[str]:
    """Обход категорий в ширину с ограничением глубины для каждого корня; статьи — без повторов."""
    wiki = Wiki(lang)
    seen_cats: set[str] = set()
    titles: dict[str, str] = {}
    per_root: Counter[str] = Counter()
    for root, max_depth in ROOTS[lang].items():
        queue = deque([(root, 0)])
        while queue:
            cat, depth = queue.popleft()
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            for m in wiki.members(cat):
                if m["ns"] == 14:
                    sub = m["title"].split(":", 1)[1]
                    if depth < max_depth and not SKIP_CATEGORY.search(sub):
                        queue.append((sub, depth + 1))
                elif not SKIP_TITLE.search(m["title"]) and m["title"] not in titles:
                    titles[m["title"]] = root
                    per_root[root] += 1
        print(f"  {root:35s} +{per_root[root]:5d} (total {len(titles)}, categories {len(seen_cats)})", flush=True)
    return list(titles)


_FORMULA = re.compile(r"\{\\displaystyle\s*")


def strip_formulas(text: str) -> str:
    """{\\displaystyle …} из plain-text выгрузки: короткие формулы оставляем без обёртки, длинные убираем."""
    out, i = [], 0
    for m in _FORMULA.finditer(text):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        inner = text[m.end():j - 1].strip()
        out.append(text[i:m.start()])
        out.append(inner if len(inner) <= 30 else "")
        i = j
    out.append(text[i:])
    return "".join(out)


def clean_wiki(text: str) -> str:
    m = TAIL_SECTIONS.search(text)
    if m:
        text = text[:m.start()]
    text = strip_formulas(text)
    text = re.sub(r"^==+\s*(.*?)\s*==+\s*$", r"\1", text, flags=re.M)  # заголовок раздела — отдельной строкой
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# порядок важен только для воспроизводимости; own — карточки и объяснения из train-части SFT
CORPUS_FILES = ("wiki_en.jsonl", "wiki_ru.jsonl", "facure.jsonl", "own.jsonl")
VAL_SHARE = 0.02


def split_of(doc: dict) -> str:
    """Деление по документам, а не по окнам: статья целиком либо в train, либо в val."""
    key = f"{doc['source']}|{doc['lang']}|{doc['title']}".encode()
    return "val" if int(hashlib.md5(key).hexdigest(), 16) % 10_000 < VAL_SHARE * 10_000 else "train"


def corpus_docs(split: str | None = None, files: tuple[str, ...] = CORPUS_FILES) -> Iterator[dict]:
    for name in files:
        for doc in read_jsonl(CORPUS / name):
            if split is None or split_of(doc) == split:
                yield doc


def read_jsonl(path: Path) -> Iterator[dict]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def fetch_wiki(lang: str, workers: int, min_chars: int = 500) -> None:
    titles = json.loads((CORPUS / f"wiki_{lang}_titles.json").read_text(encoding="utf-8"))
    out = CORPUS / f"wiki_{lang}.jsonl"
    skipped_path = CORPUS / f"wiki_{lang}_skipped.txt"  # короткие и пропавшие статьи: второй раз не качаем
    done = {d["title"] for d in read_jsonl(out)}
    if skipped_path.exists():
        done |= set(skipped_path.read_text(encoding="utf-8").splitlines())
    todo = [t for t in titles if t not in done]
    print(f"{lang}: {len(titles)} titles, {len(done)} done, {len(todo)} to fetch", flush=True)
    wiki = Wiki(lang)
    t0, n_ok = time.time(), 0
    with out.open("a", encoding="utf-8") as f, skipped_path.open("a", encoding="utf-8") as skipped, \
            ThreadPoolExecutor(workers) as pool:
        failed = 0
        for i, (title, raw) in enumerate(zip(todo, pool.map(wiki.try_extract, todo)), 1):
            if isinstance(raw, Exception):
                failed += 1
                print(f"  skip for now: {raw}", flush=True)
                continue
            text = clean_wiki(raw) if raw else ""
            if len(text) >= min_chars:
                f.write(json.dumps({"source": "wikipedia", "lang": lang, "title": title, "text": text},
                                   ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                skipped.write(title + "\n")
            if i % 500 == 0:
                f.flush()
                print(f"  {i}/{len(todo)} fetched, {n_ok} kept, {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"{lang}: kept {n_ok} of {len(todo)}, {failed} failed (rerun to retry) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


FACURE_REPO = "matheusfacure/python-causality-handbook"
FACURE_DIR = "causal-inference-for-the-brave-and-true"


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)              # картинки
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # ссылки → текст ссылки
    text = re.sub(r"<[^>]+>", "", text)                            # html
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: m.group(1) if len(m.group(1)) <= 30 else "", text, flags=re.S)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)                 # заголовки — обычной строкой
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_facure() -> list[dict]:
    """Текстовые ячейки глав «Causal Inference for the Brave and True» (MIT). Код не берём — модели нужен язык."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    listing = session.get(f"https://api.github.com/repos/{FACURE_REPO}/contents/{FACURE_DIR}", timeout=30)
    listing.raise_for_status()
    docs = []
    for item in listing.json():
        if not item["name"].endswith(".ipynb"):
            continue
        nb = session.get(item["download_url"], timeout=60).json()
        text = "\n\n".join("".join(c["source"]) for c in nb.get("cells", []) if c.get("cell_type") == "markdown")
        text = clean_markdown(text)
        if text:
            docs.append({"source": "facure", "lang": "en", "title": item["name"].removesuffix(".ipynb"), "text": text})
    return docs


def own_texts(max_sft: int) -> list[dict]:
    """Карточки методов прозой (EN и RU) и объяснения с вопросами из train-части SFT: val и test не трогаем."""
    from advisor.knowledge import load_cards
    from model.train_tokenizer import SFT_DIR
    from model.train_tokenizer import read_jsonl as read_sft

    cards = load_cards()
    docs = []
    for card in cards.values():
        parts = [f"{card.name}. {card.summary}", "Assumptions: " + "; ".join(card.assumptions) + "."]
        parts += [f"It works well when {why}." for why in card.good_flags.values()]
        parts += [f"It struggles when {why}." for why in card.bad_flags.values()]
        parts += [f"Prefer it over {cards[o].name if o in cards else o} when {why}." for o, why in card.prefer_over.items()]
        docs.append({"source": "cards", "lang": "en", "title": card.id, "text": "\n".join(parts)})
        if card.summary_ru:
            ru = [f"{card.name_ru}. {card.summary_ru}"]
            if card.assumptions_ru:
                ru.append("Допущения: " + "; ".join(card.assumptions_ru) + ".")
            docs.append({"source": "cards", "lang": "ru", "title": card.id, "text": "\n".join(ru)})
    seen: set[str] = set()
    for rec in read_sft(SFT_DIR / "train.jsonl", max_sft):
        text = rec["question"] + "\n" + rec["explanation"]
        if text not in seen:
            seen.add(text)
            docs.append({"source": "sft_explanations", "lang": rec["lang"], "title": rec["id"], "text": text})
    return docs


def write_docs(docs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(docs)} documents, {sum(len(d['text']) for d in docs) / 1e6:.2f}M chars → {path.relative_to(ROOT)}")


def stats() -> None:
    print(f"{'file':28s} {'docs':>7s} {'M chars':>8s}")
    for path in sorted(CORPUS.glob("*.jsonl")):
        n = chars = 0
        for d in read_jsonl(path):
            n += 1
            chars += len(d["text"])
        print(f"{path.name:28s} {n:7d} {chars / 1e6:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["titles", "wiki", "facure", "own", "stats"])
    ap.add_argument("--lang", choices=["en", "ru"], default="en")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-sft", type=int, default=20000, help="сколько train-примеров SFT взять в корпус")
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)

    if args.command == "titles":
        titles = crawl(args.lang)
        path = CORPUS / f"wiki_{args.lang}_titles.json"
        path.write_text(json.dumps(titles, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{len(titles)} titles → {path.relative_to(ROOT)}")
    elif args.command == "wiki":
        fetch_wiki(args.lang, args.workers)
    elif args.command == "facure":
        write_docs(fetch_facure(), CORPUS / "facure.jsonl")
    elif args.command == "own":
        write_docs(own_texts(args.max_sft), CORPUS / "own.jsonl")
    else:
        stats()


if __name__ == "__main__":
    main()
