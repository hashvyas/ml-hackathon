"""Convert report/report.md to report/report.pdf with reportlab.

Supports only what the report uses: # headings, paragraphs, - bullets, | tables |,
![caption](path) images, **bold** and `code`. Run from the repo root: python report/build_pdf.py
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, ListFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SRC, OUT = Path("report/report.md"), Path("report/report.pdf")

# Palatino Linotype from Windows, embedded in the PDF; Times if it is not installed
FONTS = Path("C:/Windows/Fonts")
if (FONTS / "pala.ttf").is_file():
    for name, file in [("Pala", "pala.ttf"), ("Pala-Bold", "palab.ttf"), ("Pala-Italic", "palai.ttf"), ("Pala-BoldItalic", "palabi.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
    registerFontFamily("Pala", normal="Pala", bold="Pala-Bold", italic="Pala-Italic", boldItalic="Pala-BoldItalic")
    REGULAR, BOLD, ITALIC = "Pala", "Pala-Bold", "Pala-Italic"
else:
    REGULAR, BOLD, ITALIC = "Times-Roman", "Times-Bold", "Times-Italic"

styles = getSampleStyleSheet()
for name in ("BodyText", "Heading1", "Heading2", "Heading3", "Title"):  # plain black text throughout
    styles[name].fontName = BOLD if name != "BodyText" else REGULAR
    styles[name].textColor = colors.black
body = styles["BodyText"]
body.fontSize, body.leading = 10.5, 14
small = body.clone("small", fontSize=8, leading=10)
caption = body.clone("caption", fontName=ITALIC, fontSize=9, leading=11.5, alignment=1)
author = body.clone("author", fontSize=11, leading=14, alignment=1)
WIDTH = A4[0] - 3 * cm


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)


def table(lines, plain=False):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[\s:|-]+\|$", l.strip())]
    if plain:  # author block: names over SRNs, no lines
        return Table([[Paragraph(inline(c), author) for c in r] for r in rows], colWidths=[WIDTH / len(rows[0])] * len(rows[0]))
    cells = [[Paragraph(("<b>%s</b>" if i == 0 else "%s") % inline(c), small) for c in r] for i, r in enumerate(rows)]
    # column width follows its longest word, so words are not split across lines
    weights = [max(11, max(len(r[j]) for r in rows) / 3, 1.3 * max(len(w) for r in rows for w in (r[j].split() or [""]))) for j in range(len(rows[0]))]
    t = Table(cells, colWidths=[WIDTH * w / sum(weights) for w in weights], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),  # grey header row
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def image(path, text):
    img = Image(str(path))
    scale = min(WIDTH / img.imageWidth, 11 * cm / img.imageHeight, 1.0)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return [img, Paragraph(inline(text), caption), Spacer(1, 8)]


def build():
    story, lines, i = [], SRC.read_text(encoding="utf-8").splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") and not story:  # first heading is the title
            story.append(Paragraph(inline(line.lstrip("#").strip()), styles["Title"]))
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            story.append(Paragraph(inline(line.lstrip("#").strip()), styles[f"Heading{level}"]))
        elif line.strip() == "<!-- plain -->":
            block, i = [], i + 1
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            story += [table(block, plain=True), Spacer(1, 12)]
            continue
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            story += [table(block), Spacer(1, 6)]
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(Paragraph(inline(lines[i][2:]), body)); i += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=12, bulletFontName=REGULAR))
            continue
        elif m := re.match(r"!\[(.*)\]\((.+)\)", line):
            path = (SRC.parent / m.group(2)).resolve()
            story += image(path, m.group(1)) if path.is_file() else [Paragraph(f"[missing image {m.group(2)}]", body)]
        elif line.strip():
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||- |!\[)", lines[i + 1]):
                i += 1; para.append(lines[i])
            story.append(Paragraph(inline(" ".join(para)), body))
        i += 1
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="PathSnap report").build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
