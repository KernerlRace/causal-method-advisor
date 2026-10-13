# Causal Method Advisor

По CSV и вопросу на естественном языке подсказывает методологию причинного анализа.
Ядро — своя языковая модель, написанная и обученная с нуля.

Подробный план в [PLAN.md](PLAN.md). Вопросы к преподавателю (на английском) — в [Questions_for_teacher.pdf](Questions_for_teacher.pdf).

## Установка (Windows, NVIDIA GPU)

Нужен [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/check_env.py
```

Последняя команда проверяет библиотеки, видимость GPU и прогоняет короткое обучение в fp16.

## Документы

PDF собираются из `docs/*.md`:

```bash
.venv/Scripts/python.exe scripts/md_to_pdf.py docs/questions_for_teacher_en.md Questions_for_teacher.pdf
```

## Этап 1: профиль, диагностики, правила

Запрос (что воздействие, что исход) пока задаётся вручную; позже его будет выдавать модель.

```bash
.venv/Scripts/python.exe scripts/demo_stage1.py data/demo/smoking.csv --treatment california --outcome cigsale
.venv/Scripts/python.exe -m pytest -q tests
```

- `knowledge/methods/*.yaml` — 22 карточки методов (требования, допущения, аргументы за и против).
- `advisor/profiler.py` — структура CSV: cross-section / panel / time series, дизайн воздействия в панели.
- `advisor/diagnostics.py` — параллельные тренды, синтетический контроль, overlap и баланс, поиск переменной для RDD, сила инструмента, негауссовость.
- `advisor/rules.py` — какие методы допустимы, какие с оговорками, какие запрещены и почему.

## Этап 2: генераторы сценариев

17 типов сценариев (эксперименты, back-door, IV, front-door, RDD, DiD, SC/SDID, поэтапное внедрение,
временной ряд, поиск структуры, неидентифицируемые и «не причинные» вопросы) × 7 предметных тем.
Для каждого известен правильный ответ: допустимые методы с весами, режим ответа и истинный эффект.

```bash
.venv/Scripts/python.exe -m datagen.build --n 2000 --workers 6 --save-csv 100
```

- `datagen/scenarios/` — генераторы, `datagen/naming.py` — темы и имена колонок.
- `advisor/estimators.py` — эталонные оценщики (DiD, SC, SDID, IV, RDD, AIPW, DML, ITS, ...);
  на Prop 99 совпадают с Facure: DiD −27.35, SC −19.51, SDID −15.61.
- `tests/test_stage2.py` — правила не запрещают правильные методы, диагностики видят заложенное,
  правильный метод восстанавливает эффект, а наивный ошибается сильнее.

## Этап 3: вопросы, ответы и обучающий набор

- `datagen/questions.py` — вопросы пользователя (EN основной, RU ~15%): ~120 шаблонов основного вопроса,
  контекст, которого нет в данных (рандомизация, скрытый фактор, инструмент, медиатор, порог), опечатки,
  разговорный стиль. ~10% шаблонов отложены только для теста.
- `datagen/explanations.py` — целевой ответ: разбор вопроса (проход 1), текст диагностик (вход прохода 2),
  методы с весами, режим ответа и объяснение. Все числа в объяснении берутся из диагностик.
- `datagen/build_sft.py` — сборка train/val/test (разбиение по сценариям) + `docs/sft_review_sample.md`
  с 50 примерами для ручной проверки.
- Эталон согласуется с результатом диагностик (`reconcile_gold`): если тест показал непараллельные тренды,
  правильным ответом считается SDID, даже если механизм порождения задумывался «под DiD».

```bash
.venv/Scripts/python.exe -m datagen.build --n 50000 --seed 1 --workers 9
.venv/Scripts/python.exe -m datagen.build_sft --per-scenario 2
```

## Этап 4: бейзлайн без нейросети и интерфейс

```bash
.venv/Scripts/python.exe scripts/train_baseline.py --max-parser 40000
.venv/Scripts/python.exe -m eval.run_eval --backend baseline
.venv/Scripts/streamlit run app.py
```

- `advisor/baseline.py` — разбор вопроса (TF-IDF + логрегрессия, бустинг по парам «вопрос × колонка»)
  и ранжирование методов (по модели на метод, предсказывает вес в эталоне).
- `advisor/pipeline.py` — сквозной `advise(df, question, backend)`; бэкенд подключаемый.
- `app.py` — веб-интерфейс: CSV или демо → вопрос → разбор (роли можно поправить) → совет,
  проверки допущений, карточки методов, таблица правил.
- `data/demo/` — 12 демо-датасетов с вопросами (`demos.json`), включая реальный Prop 99.

Бейзлайн на test (4970 вопросов, `eval/results/baseline_test.json`):

| метрика | бейзлайн | только правила |
|---|---|---|
| лучший метод при верном разборе (top-1) | 99.9% | 69.1% |
| сквозная top-1 | 83.6% | — |
| воздействие и исход найдены верно | 85.4% (RU 96.6%) | — |
| тип вопроса / на незнакомых шаблонах | 98.2% / 78.9% | — |

Слабые места бейзлайна — синонимы («salary» ↔ `annual_pay`) и незнакомые формулировки.
Их и должна закрыть своя модель.
