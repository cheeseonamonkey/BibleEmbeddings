"""Convert paper.md to a styled, linkable PDF using ReportLab."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Table, TableStyle

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "paper.md"
DESTINATION = BASE / "paper.pdf"
FONT = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
INK = colors.HexColor("#1b2925")
GREEN = colors.HexColor("#326449")
MUTED = colors.HexColor("#5c6962")
LINE = colors.HexColor("#d8e0d9")
PALE = colors.HexColor("#f3f7f3")
WIDTH = letter[0] - 96

def style(name, size, leading, color=INK, bold=False, before=0, after=8, indent=0, first=0, keep=False):
    return ParagraphStyle(name, fontName="DejaVu-Bold" if bold else "DejaVu",
                          fontSize=size, leading=leading, textColor=color,
                          spaceBefore=before, spaceAfter=after, leftIndent=indent,
                          firstLineIndent=first, keepWithNext=keep, splitLongWords=True)

STYLES = {
    "title": style("title", 22, 27, bold=True, after=10),
    "subtitle": style("subtitle", 11, 17, color=GREEN, after=12),
    "h2": style("h2", 12.1, 16.5, bold=True, before=13, after=6, keep=True),
    "h3": style("h3", 10, 14, color=GREEN, bold=True, before=10, after=5, keep=True),
    "body": style("body", 8.5, 13.5, after=7.5),
    "list": style("list", 8.3, 13, after=4, indent=15, first=-13),
    "thead": style("thead", 7.1, 10.2, bold=True),
    "td": style("td", 7.1, 10.2),
}

def inline(value: str) -> str:
    links = []
    def stash(m):
        links.append('<link href="' + html.escape(m.group(2), quote=True) +
                     '" color="#326449">' + html.escape(m.group(1)) + '</link>')
        return "XXLINK" + str(len(links) - 1) + "XX"
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]*)\)", stash, value)
    value = html.escape(value, quote=False)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(chr(96) + "([^" + chr(96) + "]+)" + chr(96), r'<font color="#326449">\1</font>', value)
    for i, link in enumerate(links):
        value = value.replace("XXLINK" + str(i) + "XX", link)
    return value

def build():
    doc = SimpleDocTemplate(str(DESTINATION), pagesize=letter, leftMargin=48,
                            rightMargin=48, topMargin=41, bottomMargin=44,
                            title="Exhibit A: The Burning Bush — Verse Lawyer",
                            author="BibleEmbeddings", subject="Biblical evidence retrieval and NLI")
    story, pending = [], []
    def flush():
        if pending:
            story.append(Paragraph(inline(" ".join(pending)), STYLES["body"]))
            pending.clear()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    seen_section = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
            i += 1
            continue
        if line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                fields = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not (len(fields) > 1 and all(re.fullmatch(r":?-+:?", c) for c in fields)):
                    rows.append(fields)
                i += 1
            count = len(rows[0])
            if any(len(row) != count for row in rows):
                raise ValueError("Table has inconsistent columns")
            weights = ([.44, .19, .19, .18] if count == 4 else
                       [.48, .31, .21] if count == 3 else [1 / count] * count)
            grid = [[Paragraph(inline(cell), STYLES["thead" if rowno == 0 else "td"])
                     for cell in row] for rowno, row in enumerate(rows)]
            t = Table(grid, colWidths=[WIDTH * x for x in weights], repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), .8, LINE),
                ("LINEBELOW", (0, 1), (-1, -2), .3, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            t.spaceAfter = 10
            story.append(t)
            continue
        if line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            kind = "title" if level == 1 else ("subtitle" if level == 3 and not seen_section else
                   "h2" if level == 2 else "h3")
            story.append(Paragraph(inline(line[level:].strip()), STYLES[kind]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=1.1, color=GREEN, spaceAfter=13))
            if level == 2:
                seen_section = True
            i += 1
            continue
        if re.match(r"^(?:\d+[.]|[-*])\s+", line):
            flush()
            story.append(Paragraph(inline(line), STYLES["list"]))
            i += 1
            continue
        pending.append(line)
        i += 1
    flush()
    def footer(canvas, current):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(48, 38, letter[0] - 48, 38)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, 26, "VERSE LAWYER   /   RESEARCH NOTE   /   2026")
        canvas.drawRightString(letter[0] - 48, 26, str(current.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(str(DESTINATION), DESTINATION.stat().st_size, "bytes")

if __name__ == "__main__":
    build()
