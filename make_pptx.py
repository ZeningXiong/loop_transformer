"""把 docs/REPORT.md（= REPORT.docx）的内容做成 PPT：docs/REPORT.pptx

设计：16:9、深蓝主色、每页一个结论；数字优先、少字。
"""
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

# ---------- 主题 ----------
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x4C, 0x78, 0xA8)
RED = RGBColor(0xE4, 0x57, 0x56)
GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def cjk(run, size=None, bold=None, color=None, font=FONT):
    run.font.name = font
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", font)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def textbox(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            space_after=6, line_spacing=1.15):
    """lines: [(text, size, bold, color, indent_level), ...]"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    first = True
    for item in lines:
        text, size, bold, color = item[0], item[1], item[2], item[3]
        level = item[4] if len(item) > 4 else 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.level = level
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = text
        cjk(r, size, bold, color)
    return tb


def title_bar(slide, title, kicker=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), W, Inches(0.14))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    y = Inches(0.42)
    if kicker:
        textbox(slide, Inches(0.65), y, Inches(11.8), Inches(0.3),
                [(kicker, 12, True, BLUE)])
        y = Inches(0.72)
    textbox(slide, Inches(0.65), y, Inches(11.8), Inches(0.7),
            [(title, 26, True, NAVY)])
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), y + Inches(0.72),
                                  Inches(12.0), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xD9, 0xE1, 0xEC)
    line.line.fill.background()


def footer(slide, n, total, note="SenseNova-Loop-9B · 8 GB 单卡推理优化"):
    textbox(slide, Inches(0.65), Inches(7.02), Inches(9), Inches(0.3),
            [(note, 9, False, RGBColor(0x99, 0x99, 0x99))])
    textbox(slide, Inches(11.9), Inches(7.02), Inches(0.8), Inches(0.3),
            [(f"{n}/{total}", 9, False, RGBColor(0x99, 0x99, 0x99))], align=PP_ALIGN.RIGHT)


def add_table(slide, x, y, w, data, col_w=None, font_size=12, header_size=12,
              row_h=Inches(0.42), header_fill=NAVY, first_col_bold=False, align_right=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, x, y, w, row_h * rows)
    table = shape.table
    if col_w:
        total = sum(col_w)
        for i, cwid in enumerate(col_w):
            table.columns[i].width = Emu(int(w * cwid / total))
    for r in range(rows):
        table.rows[r].height = row_h
        for c in range(cols):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            if align_right and c in align_right:
                p.alignment = PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(data[r][c])
            cjk(run, header_size if r == 0 else font_size,
                True if (r == 0 or (first_col_bold and c == 0)) else False,
                WHITE if r == 0 else GRAY)
    return table


def kpi(slide, x, y, w, h, value, label, sub=None, color=BLUE):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = RGBColor(0xD9, 0xE1, 0xEC)
    box.adjustments[0] = 0.06
    textbox(slide, x + Inches(0.18), y + Inches(0.14), w - Inches(0.36), Inches(0.6),
            [(value, 24, True, color)], align=PP_ALIGN.LEFT)
    textbox(slide, x + Inches(0.18), y + Inches(0.78), w - Inches(0.36), Inches(0.5),
            [(label, 12, True, GRAY)])
    if sub:
        textbox(slide, x + Inches(0.18), y + Inches(1.18), w - Inches(0.36), Inches(0.5),
                [(sub, 10, False, RGBColor(0x88, 0x88, 0x88))])


slides = []


def new_slide(title=None, kicker=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        title_bar(s, title, kicker)
    slides.append(s)
    return s


# ---------- 1 封面 ----------
s = prs.slides.add_slide(BLANK)
slides.append(s)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = WHITE
bg.line.fill.background()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(2.05), Inches(0.10), Inches(2.0))
band.fill.solid()
band.fill.fore_color.rgb = BLUE
band.line.fill.background()
textbox(s, Inches(0.95), Inches(1.95), Inches(11.4), Inches(1.7),
        [("在 8 GB 笔记本显卡上", 38, True, NAVY),
         ("跑 9.4B 循环语言模型", 38, True, NAVY)],
        line_spacing=1.15)
textbox(s, Inches(0.95), Inches(3.95), Inches(11.4), Inches(0.7),
        [("项目报告：背景 · 模型 · 优化历程 · 前景", 19, False, BLUE)])
textbox(s, Inches(0.95), Inches(4.95), Inches(11.4), Inches(1.0),
        [("SenseNova-Loop-9B-Preview（9.4B，24 层循环两次 = 56 次层执行/token）", 13, False, GRAY),
         ("硬件：RTX 4070 Laptop · 8 GB 显存 · 32 GB 内存 · Windows", 13, False, GRAY)])

# ---------- 2 一页速览 ----------
s = new_slide("一页速览", "RESULTS AT A GLANCE")
kpi(s, Inches(0.65), Inches(1.65), Inches(2.9), Inches(1.75), "跑起来了", "能否运行",
    "起点：需要 >9 GB 显存")
kpi(s, Inches(3.75), Inches(1.65), Inches(2.9), Inches(1.75), "20.2 tok/s", "16K 上下文",
    "纯贪心只有 1.7", color=RED)
kpi(s, Inches(6.85), Inches(1.65), Inches(2.9), Inches(1.75), "12×", "端到端加速",
    "相对朴素实现")
kpi(s, Inches(9.95), Inches(1.65), Inches(2.7), Inches(1.75), "5.4 GB", "模型自身显存",
    "整卡 6.8–7.4 GB")
textbox(s, Inches(0.65), Inches(3.75), Inches(12.0), Inches(0.5),
        [("三条主线把“跑不起来”变成“可交互”", 15, True, NAVY)])
add_table(s, Inches(0.65), Inches(4.25), Inches(12.0), [
    ["维度", "起点", "现在", "关键手段"],
    ["生成速度", "1.7 tok/s", "20.2 tok/s（16K）· 23.5（8K）", "图 + 投机 + 量化 KV 注意力"],
    ["上下文", "2K", "16K 稳定 / 24K 可跑", "固定槽位 KV + 分层量化"],
    ["显存", "9.04 GB（超限）", "模型自身 5.4 GB", "NF4 + KV int8/int4 + 融合内核"],
    ["质量", "bf16 基准", "KL 0.146；开思考数学 8/8", "逐位一致的融合内核"],
], col_w=[2.0, 2.6, 4.2, 4.0], font_size=11.5, row_h=Inches(0.52))

# ---------- 3 背景：为什么装不下 ----------
s = new_slide("为什么原本跑不起来", "1 · 项目背景")
add_table(s, Inches(0.65), Inches(1.6), Inches(8.6), [
    ["项目", "需求", "8 GB 卡的现实"],
    ["权重（bf16）", "约 19 GB", "差 11 GB"],
    ["权重（4-bit / int8 混合）", "约 5.7 GB", "只剩 2.9 GB 给 KV、激活、显存池"],
    ["KV cache（16K，bf16）", "约 0.9 GB", "分层量化后 0.36 GB"],
    ["激活 / 显存池 / CUDA 上下文", "1–2 GB", "—"],
], col_w=[3.2, 1.9, 3.5], font_size=11.5, row_h=Inches(0.52))
textbox(s, Inches(9.55), Inches(1.6), Inches(3.2), Inches(3.2),
        [("而且它比同参数量模型“贵” 1.75×", 13, True, RED),
         ("32 个物理层，其中 24 层要循环执行两次", 11.5, False, GRAY),
         ("→ 每 token 56 次层执行", 11.5, True, NAVY),
         ("所以问题不是“装不装下”，而是“装下后跑不跑得动”。", 11.5, False, GRAY)],
        line_spacing=1.2)

band3 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.55),
                           Inches(12.0), Inches(1.05))
band3.fill.solid()
band3.fill.fore_color.rgb = LIGHT
band3.line.color.rgb = RGBColor(0xD9, 0xE1, 0xEC)
band3.adjustments[0] = 0.08
textbox(s, Inches(0.95), Inches(5.78), Inches(11.4), Inches(0.7),
        [("1.75× 的来源：32 个物理层里 24 层执行两遍 → 4 + 24×2 + 4 = 56 次层执行/token",
          14, True, NAVY),
         ("同参数量的普通 32 层模型只需 32 次 —— 所以“装下”不等于“跑得动”。", 11.5, False, GRAY)],
        line_spacing=1.2)

# ---------- 4 项目定义 ----------
s = new_slide("把项目定义成可测量的系统问题", "1 · 项目背景")
add_table(s, Inches(0.65), Inches(1.65), Inches(6.2), [
    ["目标", "口径"],
    ["放得下", "模型自身 ≤6 GB，16K 上下文"],
    ["跑得动", "≥10 tok/s（目标 20）"],
    ["不牺牲质量", "KL ≤0.15；数学题正确率不变"],
    ["可复现", "每一步优化都有对照实验"],
], col_w=[2.2, 4.0], font_size=12, row_h=Inches(0.55))
textbox(s, Inches(7.25), Inches(1.6), Inches(5.4), Inches(0.4),
        [("五条测量纪律（都是踩坑换来的）", 14, True, NAVY)])
textbox(s, Inches(7.25), Inches(2.05), Inches(5.4), Inches(3.6),
        [("1. 所有测速先预热（冷启动慢 2.8×）", 12, False, GRAY),
         ("2. 一进程一配置，并记录整卡峰值", 12, False, GRAY),
         ("3. 8.3 GB 是分页悬崖：越线静默掉到 1/4 速度", 12, False, GRAY),
         ("4. 单次数字不可信：连跑 3 次取中位", 12, False, GRAY),
         ("5. 必须在真实组合下测（eager 下可能完全测不出收益）", 12, False, GRAY)],
        space_after=10)

# ---------- 5 模型基本信息 ----------
s = new_slide("模型：SenseNova-Loop-9B", "2 · 模型")
add_table(s, Inches(0.65), Inches(1.6), Inches(7.4), [
    ["项", "值"],
    ["参数量", "9.4B（全部激活，非 MoE）"],
    ["物理层数 / 每 token 执行", "32 层 / 56 次"],
    ["注意力结构", "8 层 full attention + 24 层 Gated DeltaNet"],
    ["隐藏维度 / 头", "4096 / 16 查询头、4 KV 头、head_dim 256"],
    ["形态", "decoder-only（多模态外壳，只取文本解码器）"],
], col_w=[2.6, 4.8], font_size=11.5, row_h=Inches(0.56))
textbox(s, Inches(8.35), Inches(1.6), Inches(4.3), Inches(3.6),
        [("三个非常规设计", 14, True, NAVY),
         ("① 层循环：第 4–27 层执行两遍 → 深度翻倍、参数不变", 11.5, False, GRAY),
         ("② mHC 路由：循环段内 2 条并行流 + Sinkhorn 归一化", 11.5, False, GRAY),
         ("③ 混合注意力：线性层省显存，全局层保召回", 11.5, False, GRAY)],
        space_after=10, line_spacing=1.25)

# ---------- 6 56 次层执行 ----------
s = new_slide("56 次层执行：循环把“深度”从存储搬到时间", "2 · 模型")
blocks = [(Inches(0.9), "第 0–3 层", "执行 1 次", LIGHT, NAVY),
          (Inches(4.15), "第 4–27 层", "执行 2 次（×2）", RGBColor(0xDC, 0xE6, 0xF3), BLUE),
          (Inches(8.6), "第 28–31 层", "执行 1 次", LIGHT, NAVY)]
for x, t1, t2, fill, col in blocks:
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.95), Inches(2.85), Inches(1.35))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = RGBColor(0xC5, 0xD3, 0xE6)
    textbox(s, x + Inches(0.15), Inches(2.2), Inches(2.55), Inches(0.4), [(t1, 16, True, col)],
            align=PP_ALIGN.CENTER)
    textbox(s, x + Inches(0.15), Inches(2.7), Inches(2.55), Inches(0.4), [(t2, 13, False, GRAY)],
            align=PP_ALIGN.CENTER)
textbox(s, Inches(0.9), Inches(3.55), Inches(11.6), Inches(0.5),
        [("4 + 24×2 + 4 = 56 次层执行/token（同参数量普通模型约 32 次 → 1.75× 计算）", 14, True, NAVY)],
        align=PP_ALIGN.CENTER)
add_table(s, Inches(0.9), Inches(4.35), Inches(11.6), [
    ["层类型", "每 token 执行次数", "是否占 KV", "单次耗时（16K）"],
    ["full attention（第 3/7/…/31 层）", "14", "是（14 个槽位）", "3.74 ms"],
    ["Gated DeltaNet（线性注意力）", "42", "否（O(1) 状态）", "1.46 ms"],
], col_w=[4.2, 2.4, 2.6, 2.4], font_size=11.5, row_h=Inches(0.5))

# ---------- 7 Loop vs 蒸馏 ----------
s = new_slide("Loop Transformer 与知识蒸馏的区别", "2 · 模型")
add_table(s, Inches(0.65), Inches(1.6), Inches(12.0), [
    ["", "知识蒸馏（KD）", "层循环（Loop）"],
    ["做法", "大模型（教师）教小模型（学生）", "同一批参数反复执行"],
    ["省下的是", "参数 / 显存", "参数 / 显存"],
    ["花掉的是", "一次蒸馏训练成本", "每 token 的计算与延迟"],
    ["推理速度", "学生模型的速度（1× 深度）", "∝ 循环次数（本项目 1.75×）"],
    ["能力来源", "教师能力被压缩进学生", "同一份能力“多用几遍”"],
], col_w=[2.0, 5.0, 5.0], font_size=12, row_h=Inches(0.56))
textbox(s, Inches(0.65), Inches(5.35), Inches(12.0), Inches(1.2),
        [("蒸馏是“把大模型压小”，循环是“把小模型用多次”。两者正交、可叠加。",
          16, True, NAVY),
         ("定量上，一次循环折算的“等效独立参数”小于一层真参数（410M 循环 ≈ 580M 非循环，但训练成本 ≈ 1B）",
          11.5, False, GRAY)], line_spacing=1.25)

# ---------- 8 优化总览 ----------
s = new_slide("优化历程：两条线，各自解决一个问题", "3 · 优化历程")
kpi(s, Inches(0.65), Inches(1.7), Inches(5.85), Inches(1.6), "38 GB → 4.5 GB", "容量线：权重 NF4",
    "KV bf16 897 MiB → 分层量化 360 MiB")
kpi(s, Inches(6.8), Inches(1.7), Inches(5.85), Inches(1.6), "1.7 → 20.2 tok/s", "速度线：三次跃迁",
    "折叠注意力 → 图+投机 → 量化 KV 注意力", color=RED)
add_table(s, Inches(0.65), Inches(3.6), Inches(12.0), [
    ["维度", "容量 / 内存线", "生成速度线"],
    ["要解决的问题", "8 GB 装不下 9.4B + 16K", "20 tok/s 才可交互"],
    ["第一大手段", "离线 NF4（38 GB → 4.5 GB）", "折叠 GQA 注意力（前向 −25%）"],
    ["第二大手段", "KV 分层量化（-60%）", "图 + 2-token 投机（×1.7）"],
    ["第三大手段", "int4 融合解包（17×）", "注意力直接读量化 KV（+21%）"],
    ["最终结果", "模型自身 5.4 GB", "16K 20.2 tok/s / 8K 23.5"],
], col_w=[2.4, 4.8, 4.8], font_size=11.5, row_h=Inches(0.52))

# ---------- 9 内存线 ----------
s = new_slide("容量与内存：七步把 9.4B 塞进 8 GB", "3.1 · 容量与内存")
add_table(s, Inches(0.65), Inches(1.55), Inches(12.0), [
    ["#", "措施", "效果"],
    ["1", "离线 NF4 量化权重", "38 GB → 4.5 GB（KL 0.146）"],
    ["2", "丢弃视觉塔", "−0.6 GB"],
    ["3", "固定槽位 KV + int8", "16K 的 KV 897 → 456.8 MiB"],
    ["4", "第 2 遍 KV 压到 int4", "456.8 → 360.4 MiB"],
    ["5", "融合 int4 解包内核", "每层 5.6 → 0.32 ms（17×，逐位一致）"],
    ["6", "MTP 头也量化", "−0.35 GB"],
    ["7", "显存守卫 + empty_cache", "回收约 1.1 GB 分配器碎片"],
], col_w=[0.6, 5.4, 6.0], font_size=11.5, row_h=Inches(0.46))
textbox(s, Inches(0.65), Inches(5.6), Inches(12.0), Inches(0.9),
        [("结果：16K 下模型自身 5.4 GB（权重 4.5 + KV 0.36 + 激活/图池），整卡 6.8–7.4 GB。",
          14, True, NAVY),
         ("权重也有“按层混合精度”选项：int8 混合把 KL 减半（0.146→0.078），代价是 1.2 GB 显存，16K 下没有余量。",
          11.5, False, GRAY)], line_spacing=1.2)

# ---------- 10 KV 分层 ----------
s = new_slide("KV 为什么不是“均匀压缩”，而是按敏感度分层", "3.1 · 容量与内存")
textbox(s, Inches(0.65), Inches(1.55), Inches(6.0), Inches(0.4),
        [("把两组 KV 都压到 2-bit，测 KL：", 13, True, NAVY)])
add_table(s, Inches(0.65), Inches(2.05), Inches(5.6), [
    ["KV 组别", "压到 2-bit 时的 KL"],
    ["第一遍 KV", "1.80"],
    ["第二遍 KV", "0.33–0.39"],
], col_w=[3.0, 2.6], font_size=12.5, row_h=Inches(0.55), first_col_bold=True)
textbox(s, Inches(0.65), Inches(4.0), Inches(5.6), Inches(1.6),
        [("第一遍敏感约 5 倍 → 按敏感度分配比特", 14, True, RED),
         ("第 1 遍一律 int8，第 2 遍压到 int4：", 12, False, GRAY),
         ("456.8 → 360.4 MiB，精度与全 int8 同级", 12, True, NAVY)], line_spacing=1.25)
add_table(s, Inches(6.7), Inches(1.55), Inches(5.9), [
    ["失败尝试", "结果"],
    ["KV 压到 2-bit（LoopVQ 式）", "末段 KL 2.19，长文崩"],
    ["跨层 / 首访共享 K", "代码 prompt 15–22 token 分叉"],
    ["第一遍也压 4-bit", "直接崩（敏感 5×）"],
    ["权重全 int8", "16K 整卡 100%，没余量跑图"],
], col_w=[2.9, 3.0], font_size=11, row_h=Inches(0.62), header_fill=RED)

# ---------- 11 速度线 ----------
s = new_slide("生成速度：八项优化，最大一笔是“启用一个已经写好的东西”", "3.2 · 生成速度")
add_table(s, Inches(0.65), Inches(1.55), Inches(12.0), [
    ["#", "措施", "效果"],
    ["1", "启用折叠 GQA 注意力（代码早就有，生产路径没调）", "16K 前向 165.6 → 124.0 ms（−25%）"],
    ["2", "图（CUDA Graph）+ 2-token 投机", "16K 16.8–17.9 tok/s"],
    ["3", "注意力直接读量化 KV（自写 split-L flash 内核）", "16.80 → 20.31 tok/s（+21%）"],
    ["4", "2-token MTP 自投机", "相对贪心 ×1.4–1.67"],
    ["5", "2×2 Sinkhorn 闭式解", "一步 125 → 96.7 ms（−23%）"],
    ["6", "int8 小批量 GEMV 修复", "2-token 验证 239 → 150 ms"],
    ["7", "融合 RMSNorm + SiLU×up（逐位一致）", "16K 前向 −5%"],
    ["8", "固定 KV 缓冲（不再 torch.cat）", "免掉每步拼接，也是显存稳定的前提"],
], col_w=[0.6, 7.0, 4.4], font_size=10.5, row_h=Inches(0.44))

band11 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.9),
                            Inches(12.0), Inches(0.85))
band11.fill.solid()
band11.fill.fore_color.rgb = LIGHT
band11.line.color.rgb = RGBColor(0xD9, 0xE1, 0xEC)
band11.adjustments[0] = 0.1
textbox(s, Inches(0.95), Inches(6.08), Inches(11.4), Inches(0.5),
        [("三次跃迁：9.8 → 16.8 → 20.3 tok/s（纯贪心 → 图+投机 → 再叠量化 KV 注意力，16K）",
          13.5, True, NAVY)])

# ---------- 12 链式草稿图 ----------
s = new_slide("为什么只做 2-token：链式草稿准确率断崖", "3.2 · 生成速度")
img = os.path.join(HERE, "assets", "spec_chain.png")
s.shapes.add_picture(img, Inches(0.5), Inches(1.55), width=Inches(7.1))
textbox(s, Inches(7.95), Inches(1.6), Inches(4.9), Inches(4.8),
        [("逐位接受率", 13, True, NAVY),
         ("94% → 30% → 15% → 0%", 18, True, RED),
         ("链式到达率：94% → 28% → 4% → 0%", 12, False, GRAY),
         ("期望提交 1 + Σ = 2.26 token/步", 12, True, NAVY),
         ("", 8, False, GRAY),
         ("多一个草稿的盈亏平衡点 ≈ 75%；", 12, False, GRAY),
         ("第 2 个位置只有 30%，所以 3/4/6-token 分别慢 12%/18%/35%。", 12, False, GRAY),
         ("", 8, False, GRAY),
         ("2-token 验证是这套模型的最优点。", 13, True, NAVY)], line_spacing=1.25)

# ---------- 13 失败清单 ----------
s = new_slide("失败清单与测量纪律：这些不要重复踩", "3 · 优化历程")
add_table(s, Inches(0.65), Inches(1.6), Inches(7.3), [
    ["失败尝试", "结果"],
    ["3 / 4 / 6-token 投机", "慢 12% / 18% / 35%"],
    ["放宽验证判据（近似接受）", "只捞回 8% 失配，净收益 3%"],
    ["推理时把循环跑 4 遍", "慢 1.89×，输出退化"],
    ["把 MTP 捕获进图", "命中 15/16 → 5/16，净 −35%"],
    ["换 WSL2 / Linux", "反而慢 2–6%"],
    ["KV 2-bit / K 共享", "精度崩 / 代码分叉"],
], col_w=[3.6, 3.7], font_size=11, row_h=Inches(0.5), header_fill=RED)
textbox(s, Inches(8.35), Inches(1.6), Inches(4.3), Inches(4.6),
        [("为什么这些值得记录", 14, True, NAVY),
         ("• 每一个都曾被当作“显然有效”", 11.5, False, GRAY),
         ("• 失败原因都定位到了具体机制", 11.5, False, GRAY),
         ("• 它们共同给出了这套模型的边界：", 11.5, False, GRAY),
         ("  算力 ∝ 循环数、KV ∝ 遍数、投机 ∝ 草稿质量", 11.5, True, NAVY)],
        line_spacing=1.3, space_after=10)

# ---------- 14 应用场景 ----------
s = new_slide("Loop Transformer 的应用场景与前景", "4 · 前景")
add_table(s, Inches(0.65), Inches(1.55), Inches(7.6), [
    ["场景", "适配度 / 原因"],
    ["显存受限的端侧部署", "★★★★ 循环用参数换深度，权重更小"],
    ["长上下文但算力不宽裕", "★★★ 配合线性注意力，KV 只来自少数层"],
    ["推理时算力换能力", "★★★★ 循环次数是可调旋钮"],
    ["改造已有 checkpoint", "★★★ 不需要从头训练"],
    ["硬实时 / 大 batch 服务", "★ 延迟与计算 ∝ 循环数"],
], col_w=[3.0, 4.6], font_size=11, row_h=Inches(0.56))
textbox(s, Inches(8.5), Inches(1.55), Inches(4.2), Inches(4.8),
        [("天花板", 14, True, NAVY),
         ("① 循环收益递减：一次循环 < 一层真参数", 11.5, False, GRAY),
         ("② 算力一定变贵（省显存不省算力）", 11.5, False, GRAY),
         ("③ 召回仍靠注意力：线性:全局 = 3:1–6:1", 11.5, False, GRAY),
         ("④ KV 随循环数增长，需 LoopKV/LooM 类压缩", 11.5, False, GRAY)],
        line_spacing=1.3, space_after=10)

band14 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.95),
                            Inches(12.0), Inches(0.8))
band14.fill.solid()
band14.fill.fore_color.rgb = LIGHT
band14.line.color.rgb = RGBColor(0xD9, 0xE1, 0xEC)
band14.adjustments[0] = 0.1
textbox(s, Inches(0.95), Inches(6.12), Inches(11.4), Inches(0.5),
        [("结论：它适合“缺显存”的端侧，不适合“缺算力/要长上下文”的场景。",
          13.5, True, NAVY)])

# ---------- 15 相关工作 ----------
s = new_slide("与 loop transformer 直接相关的工作", "5 · 相关工作")
add_table(s, Inches(0.65), Inches(1.55), Inches(12.0), [
    ["方向", "工作", "与本项目的关系"],
    ["KV 分层", "LoopKV（arXiv:2611.04137）", "同思路最简版：我们只分 2 档（第1遍 int8 / 第2遍 int4）"],
    ["KV 冗余", "LoopVQ（2611.05304）", "我们试过标量 2-bit，末段 KL 2.19 失败；他们用码本做到 0.5 bit"],
    ["KV 不随循环增长", "LooM（2611.06835）", "我们试过后验共享，代码 prompt 15–22 token 分叉"],
    ["kernel 融合", "ExactLoop（2611.05150）", "同样用舍入栅栏做逐位一致，同样拒绝近似融合"],
    ["免训练草稿", "CopyLoop（2610.38822）", "我们没做：MTP 在短上下文/代码上很弱，正是 copy 的强项"],
    ["早 loop 草稿", "LoopAhead（2610.21594）", "同为自投机，但草稿来源不同（早期循环 vs MTP 头）"],
], col_w=[2.0, 3.2, 6.8], font_size=10.5, row_h=Inches(0.62))

# ---------- 16 结尾 ----------
s = prs.slides.add_slide(BLANK)
slides.append(s)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
textbox(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(2.4),
        [("一个“理论上跑不起来”的 9.4B 循环模型，", 26, True, WHITE),
         ("在 8 GB 笔记本显卡上做到 16K / 20.2 tok/s / 5.4 GB。", 26, True, WHITE),
         ("", 10, False, WHITE),
         ("最有价值的不是速度数字，而是一套能区分“真优化”与“测量假象”的方法，", 15, False, RGBColor(0xC9, 0xD8, 0xEC)),
         ("以及 40 余次失败尝试的归档。", 15, False, RGBColor(0xC9, 0xD8, 0xEC))],
        line_spacing=1.2)

# ---------- 页脚 ----------
total = len(slides)
for i, sl in enumerate(slides[1:-1], start=2):
    footer(sl, i, total)

out = os.path.join(HERE, "docs", "REPORT.pptx")
prs.save(out)
print(f"已生成 {out}，共 {total} 页")
