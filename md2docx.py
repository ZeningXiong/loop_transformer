"""把 docs/REPORT.md 转成 docs/REPORT.docx（面向人工修改）。

支持：标题层级、表格、无序/有序列表、引用块、图片、行内 **粗体** / *斜体* / `代码` / [链接](url)。
中文字体用微软雅黑，正文 10.5pt，A4 页面。
"""
import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "docs", "REPORT.md")
DST = os.path.join(HERE, "docs", "REPORT.docx")
sys.stdout.reconfigure(encoding="utf-8")


def set_cjk(run, font="Microsoft YaHei"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = "Microsoft YaHei"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.0)
        s.left_margin = s.right_margin = Cm(2.2)


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\[[^\]]+?\]\([^)]+?\))")


def add_inline(p, text):
    """把行内标记写进段落。"""
    text = text.replace("\\*", "*")
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        elif part.startswith("[") and "](" in part:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part).groups()
            r = p.add_run(label)
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
            r.underline = True
            p.add_run(f"（{url}）").font.size = Pt(8)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            p.add_run(part)
        for r in p.runs:
            set_cjk(r)


def shade(cell, hexcolor="F2F2F2"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        add_inline(cell.paragraphs[0], h)
        for r in cell.paragraphs[0].runs:
            r.bold = True
        shade(cell)
    # 表头在跨页时重复
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for row in body:
        cells = t.add_row().cells
        for i, txt in enumerate(row[:len(header)]):
            cells[i].text = ""
            add_inline(cells[i].paragraphs[0], txt)
    # 行内不跨页
    for r in t.rows:
        trPr = r._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr.append(cant)
    doc.add_paragraph()


def is_table_line(line):
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


raw = open(SRC, encoding="utf-8").read().split("\n")
md = [re.sub(r"^>\s?", "", ln) for ln in raw]
doc = Document()
style_doc(doc)
i = 0
while i < len(md):
    line = md[i]
    stripped = line.strip()

    if not stripped:
        i += 1
        continue

    # 分隔线
    if re.fullmatch(r"-{3,}", stripped):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        pPr = p._p.get_or_add_pPr()
        pbdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "BFBFBF")
        pbdr.append(bottom)
        pPr.append(pbdr)
        i += 1
        continue

    # 标题
    m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
    if m:
        level, title = len(m.group(1)), m.group(2)
        if level == 1:
            h = doc.add_heading(level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(h, title)
        else:
            h = doc.add_heading(level={2: 1, 3: 2, 4: 3}.get(level, 3))
            add_inline(h, title)
        size = {1: 16, 2: 13, 3: 11.5}.get(min(level, 4), 11.5)
        for r in h.runs:
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            set_cjk(r)
        i += 1
        continue

    # 图片
    m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
    if m:
        alt, src = m.groups()
        path = os.path.normpath(os.path.join(os.path.dirname(SRC), src))
        if os.path.exists(path):
            doc.add_picture(path, width=Cm(15.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            doc.add_paragraph(f"[图片缺失: {src}]")
        i += 1
        continue

    # 表格
    if is_table_line(stripped) and i + 1 < len(md) and re.match(r"^\|[\s:\-|]+\|$", md[i + 1].strip()):
        rows = [split_row(stripped)]
        j = i + 2
        while j < len(md) and is_table_line(md[j].strip()):
            rows.append(split_row(md[j].strip()))
            j += 1
        add_table(doc, rows)
        i = j
        continue

    # 引用块
    if stripped.startswith(">"):
        p = doc.add_paragraph(style="Intense Quote")
        add_inline(p, stripped.lstrip("> ").strip())
        i += 1
        continue

    # 无序列表（含缩进层级）
    m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
    if m:
        indent, text = len(m.group(1)), m.group(2)
        style = "List Bullet" if indent < 2 else "List Bullet 2"
        p = doc.add_paragraph(style=style)
        add_inline(p, text)
        i += 1
        continue

    # 有序列表
    m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
    if m:
        p = doc.add_paragraph(style="List Number")
        add_inline(p, m.group(2))
        i += 1
        continue

    if stripped in ("一页速览",):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(stripped)
        r.bold = True
        r.font.size = Pt(12)
        set_cjk(r)
        i += 1
        continue

    # 普通段落
    p = doc.add_paragraph()
    add_inline(p, stripped)
    i += 1

# 页脚页码
def add_page_numbers(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for el, attr in (("w:fldChar", {"w:fldCharType": "begin"}),
                         ("w:instrText", None),
                         ("w:fldChar", {"w:fldCharType": "end"})):
            e = OxmlElement(el)
            if attr:
                for k, v in attr.items():
                    e.set(qn(k), v)
            else:
                e.set(qn("xml:space"), "preserve")
                e.text = " PAGE "
            run._r.append(e)
        run.font.size = Pt(9)


add_page_numbers(doc)
doc.save(DST)
print("已生成:", DST)
