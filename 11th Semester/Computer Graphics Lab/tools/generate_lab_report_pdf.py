#!/usr/bin/env python3
"""Small utility to render Lab2_report.md to Lab2_report.pdf using reportlab.

This generator keeps formatting simple: headings are larger bold text, paragraphs are wrapped.
If reportlab isn't installed, run `pip install reportlab`.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Frame, Image, Table
import textwrap

INPUT = "Lab2_report.md"
OUTPUT = "Lab2_report.pdf"


def _markdown_to_blocks(md_text):
    # Extremely small markdown -> block converter (headings and paragraphs only)
    lines = md_text.splitlines()
    blocks = []
    buf = []
    for line in lines:
        if line.strip() == "":
            if buf:
                blocks.append(("p", "\n".join(buf).strip()))
                buf = []
            continue
        # headings identify
        if line.startswith("#"):
            if buf:
                blocks.append(("p", "\n".join(buf).strip()))
                buf = []
            lvl = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            blocks.append(("h", (lvl, text)))
        else:
            buf.append(line)

    if buf:
        blocks.append(("p", "\n".join(buf).strip()))

    return blocks


def make_pdf(input_path=INPUT, output_path=OUTPUT):
    with open(input_path, 'r', encoding='utf-8') as f:
        md = f.read()

    blocks = _markdown_to_blocks(md)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    normal = styles['Normal']
    h1 = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=18, leading=22)
    h2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=14, leading=18)

    story = []

    for kind, data in blocks:
        if kind == 'h':
            lvl, text = data
            if lvl == 1:
                story.append(Paragraph(text, h1))
            else:
                story.append(Paragraph(text, h2))
        elif kind == 'p':
            # simple paragraph: wrap
            for paragraph in data.split('\n\n'):
                # convert triple backticks and code excerpts into monospace -> keep simple
                text = paragraph.replace('`', '')
                story.append(Paragraph(text.replace('\n', '<br/>'), normal))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Generated: {output_path}")


if __name__ == '__main__':
    make_pdf()
