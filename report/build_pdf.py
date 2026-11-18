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
from reportlab.platypus import Image, ListFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SRC, OUT = Path("report/report.md"), Path("report/report.pdf")
styles = getSampleStyleSheet()
body, small = styles["BodyText"], styles["BodyText"].clone("small", fontSize=7, leading=8.5)
WIDTH = A4[0] - 3 * cm


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[\s:|-]+\|$", l.strip())]
    cells = [[Paragraph(inline(c), small) for c in r] for r in rows]
    t = Table(cells, colWidths=[WIDTH / len(rows[0])] * len(rows[0]), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eee8")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def image(path, caption):
    img = Image(str(path))
    scale = min(WIDTH / img.imageWidth, 11 * cm / img.imageHeight, 1.0)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return [img, Paragraph(f"<i>{inline(caption)}</i>", small), Spacer(1, 6)]


def build():
    story, lines, i = [], SRC.read_text(encoding="utf-8").splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            story.append(Paragraph(inline(line.lstrip("#").strip()), styles[f"Heading{level}"]))
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
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=12))
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
