#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""11_md2docx.py -- minimal Markdown -> .docx converter for the manuscript."""
import re, sys
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def inline(p, text):
    """render **bold**, *italic*, `code`, subscripts/superscripts left as-is"""
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for s in parts:
        if not s:
            continue
        if s.startswith("**") and s.endswith("**"):
            r = p.add_run(s[2:-2]); r.bold = True
        elif s.startswith("`") and s.endswith("`"):
            r = p.add_run(s[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
        elif s.startswith("*") and s.endswith("*"):
            r = p.add_run(s[1:-1]); r.italic = True
        else:
            p.add_run(s)


def convert(src, dst):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)

    lines = open(src, encoding="utf-8").read().split("\n")
    in_refs = False
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        # table?
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) == len(hdr):
                    rows.append(cells)
                i += 1
            t = doc.add_table(rows=1, cols=len(hdr))
            t.style = "Light Grid Accent 1"
            for j, h in enumerate(hdr):
                c = t.rows[0].cells[j]
                c.text = ""
                r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(8.5)
            for row in rows:
                cells = t.add_row().cells
                for j, v in enumerate(row):
                    cells[j].text = ""
                    r = cells[j].paragraphs[0].add_run(v); r.font.size = Pt(8.5)
            doc.add_paragraph()
            continue

        if ln.startswith("# "):
            h = doc.add_heading(ln[2:], level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif ln.startswith("## "):
            doc.add_heading(ln[3:], level=1)
            in_refs = ln[3:].strip().lower().startswith("references")
        elif ln.startswith("### "):
            doc.add_heading(ln[4:], level=2)
        elif ln.startswith("#### "):
            doc.add_heading(ln[5:], level=3)
        elif ln.startswith("---") and ln.strip() == "---":
            doc.add_paragraph()
        elif ln.startswith("> "):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.4)
            inline(p, ln[2:]); p.runs[0].italic = True
        elif re.match(r"^\s*[-*]\s+", ln):
            p = doc.add_paragraph(style="List Bullet")
            inline(p, re.sub(r"^\s*[-*]\s+", "", ln))
        elif re.match(r"^\s*\d+\.\s+", ln):
            # Keep the literal "N." in the text: python-docx's "List Number"
            # style is not bound to a visible numbering definition in a fresh
            # document, so the numbers would silently disappear in Word.
            p = doc.add_paragraph()
            if in_refs:
                pf = p.paragraph_format
                pf.left_indent = Inches(0.35)
                pf.first_line_indent = Inches(-0.35)
                pf.space_after = Pt(3)
                inline(p, ln.strip())
                for r in p.runs:
                    r.font.size = Pt(10)
            else:
                inline(p, ln.strip())
        elif ln.strip():
            p = doc.add_paragraph()
            inline(p, ln)
        i += 1
    doc.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
