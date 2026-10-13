"""Минимальный конвертер наших Markdown-документов в PDF (кириллица через Arial).

Поддерживает: # и ## заголовки, строки-заголовки **...**, списки "- ",
*курсив*, **жирный**, `код`, разделитель ---, строки-примечания "*По умолчанию...*".

Использование: python scripts/md_to_pdf.py input.md output.pdf
"""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

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
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def build_story(md: str):
    story = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
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
    doc.build(build_story(md), onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
