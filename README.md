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
