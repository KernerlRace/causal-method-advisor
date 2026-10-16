"""Минимальный конвертер наших Markdown-документов в PDF (кириллица через Arial).

Поддерживает: # и ## заголовки, строки-заголовки **...**, списки "- ",
*курсив*, **жирный**, `код`, разделитель ---, строки-примечания "*По умолчанию...*",
таблицы "| ... |", блоки кода ``` и картинки ![подпись](путь относительно .md).

Использование: python scripts/md_to_pdf.py input.md output.pdf
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", FONT_DIR + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_DIR + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FONT_DIR + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", FONT_DIR + "arialbi.ttf"))
pdfmetrics.registerFontFamily(
    "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic"
)

ACCENT = colors.HexColor("#1F4E79")
NOTE_BG = colors.HexColor("#EEF3F8")
TEXT = colors.HexColor("#222222")

base = ParagraphStyle("base", fontName="Arial", fontSize=10.5, leading=14.5, textColor=TEXT,
                      alignment=TA_LEFT, spaceAfter=3)
STYLES = {
    "h1": ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=18, leading=22,
                         textColor=ACCENT, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=13.5, leading=17,
                         textColor=ACCENT, spaceBefore=10, spaceAfter=6, keepWithNext=1),
    "qtitle": ParagraphStyle("qtitle", parent=base, fontName="Arial-Bold", fontSize=11,
                             spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "body": base,
    "bullet": ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=4, spaceAfter=2),
    "note": ParagraphStyle("note", parent=base, fontName="Arial-Italic", backColor=NOTE_BG,
                           borderPadding=(4, 6, 4, 6), leftIndent=6, rightIndent=6,
                           spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#33475B")),
    "code": ParagraphStyle("code", parent=base, fontName="Courier", fontSize=8, leading=11.5, spaceAfter=0,
                           leftIndent=6, backColor=NOTE_BG),
    "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=10.5, spaceAfter=0),
    "head": ParagraphStyle("head", parent=base, fontName="Arial-Bold", fontSize=8.5, leading=10.5, spaceAfter=0),
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table(lines: list[str]) -> Table:
    """Строки Markdown-таблицы → Table; строка-разделитель |---| пропускается, первая строка — шапка."""
    rows = [cells(l) for l in lines if not re.fullmatch(r"\|?[\s:|-]+\|?", l.strip())]
    style = STYLES["cell"]
    data = [[Paragraph(inline(c), STYLES["head"] if i == 0 else style) for c in r] for i, r in enumerate(rows)]
    first = max(len(r[0]) for r in rows)
    n = len(rows[0])
    width = A4[0] - 4 * cm
    w0 = width * (0.40 if first > 25 and n > 2 else 1 / n)
    t = Table(data, colWidths=[w0] + [(width - w0) / (n - 1)] * (n - 1), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
                           ("BACKGROUND", (0, 0), (-1, 0), NOTE_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def image(path: Path) -> Image:
    img = Image(str(path))
    scale = min(1.0, (A4[0] - 4 * cm) / img.drawWidth)
    img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
    return img


def build_story(md: str, base_dir: Path = Path(".")):
    story = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if line.startswith("```"):
            while i < len(lines) and not lines[i].startswith("```"):
                story.append(Paragraph(inline(lines[i]).replace(" ", "&nbsp;") or "&nbsp;", STYLES["code"]))
                i += 1
            i += 1
            story.append(Spacer(1, 6))
            continue
        if not line:
            continue
        if line.lstrip().startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            story += [Spacer(1, 3), table(block), Spacer(1, 6)]
            continue
        m = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line.strip())
        if m:
            story += [Spacer(1, 4), image(base_dir / m.group(1)), Spacer(1, 6)]
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), STYLES["h1"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), STYLES["h2"]))
        elif line.strip() == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#B0BEC5")))
            story.append(Spacer(1, 4))
        elif line.lstrip().startswith("- "):
            story.append(Paragraph(inline(line.lstrip()[2:]), STYLES["bullet"], bulletText="•"))
        elif re.fullmatch(r"\*\*[^*]+\*\*", line.strip()):
            story.append(Paragraph(inline(line.strip()[2:-2]), STYLES["qtitle"]))
        elif re.fullmatch(r"\*[^*].*\*", line.strip()):
            story.append(Paragraph(inline(line.strip()[1:-1]), STYLES["note"]))
        else:
            story.append(Paragraph(inline(line), STYLES["body"]))
    return story


def make_footer(page_label: str):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 8.5)
        canvas.setFillColor(colors.HexColor("#78909C"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{page_label} {doc.page}")
        canvas.restoreState()
    return footer


def main(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as f:
        md = f.read()
    title = next((l[2:].strip() for l in md.splitlines() if l.startswith("# ")), "Document")
    # язык подписи страниц определяем по заголовку: кириллица → «стр.», иначе «Page»
    footer = make_footer("стр." if re.search("[а-яА-ЯёЁ]", title) else "Page")
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title)
    doc.build(build_story(md, Path(src).parent), onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
