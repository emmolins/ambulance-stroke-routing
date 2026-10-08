"""Editable (native shape) versions of the two methods figures, built with python-pptx.
Slide 1: MDP schematic. Slide 2: patient timeline. Shapes paste into Word as editable drawings."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
import re, copy

FONT = "Times New Roman"
INK, INK2, INK3 = RGBColor(0x1a, 0x1a, 0x1a), RGBColor(0x55, 0x55, 0x55), RGBColor(0x8a, 0x8a, 0x8a)
LINE = RGBColor(0x66, 0x66, 0x66)
FILL = {"field": RGBColor(0xEE, 0xEE, 0xEE), "evt": RGBColor(0xDC, 0xE6, 0xF7), "psc": RGBColor(0xFB, 0xE5, 0xD0), "nsc": RGBColor(0xDD, 0xF0, 0xDC), "lane": RGBColor(0xF6, 0xF6, 0xF6)}
RED = RGBColor(0xB0, 0x1F, 0x24); BLUE = RGBColor(0x1F, 0x4E, 0xA3)

prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(5.625)
blank = prs.slide_layouts[6]

# ---------- text helpers: "_{x}" -> subscript, "^{x}" -> superscript, "*b*" bold, "/i/" italic
TOK = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|\*[^*]+\*|/[^/]+/)")
def add_rich(tf_or_p, text, size=9, color=INK, align=None, bold=False, italic=False):
    p = tf_or_p if hasattr(tf_or_p, "runs") else tf_or_p.paragraphs[0]
    if align is not None: p.alignment = align
    for piece in TOK.split(text):
        if not piece: continue
        sub = sup = False; b = bold; it = italic
        if piece.startswith("_{"): piece = piece[2:-1]; sub = True
        elif piece.startswith("^{"): piece = piece[2:-1]; sup = True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2: piece = piece[1:-1]; b = True
        elif piece.startswith("/") and piece.endswith("/") and len(piece) > 2: piece = piece[1:-1]; it = True
        r = p.add_run(); r.text = piece; f = r.font; f.name = FONT; f.size = Pt(size); f.bold = b; f.italic = it; f.color.rgb = color
        if sub or sup: f._element.set("baseline", "-25000" if sub else "30000")
    return p

def noshadow(shp):
    st = shp._element.find(qn("p:style"))
    if st is not None: shp._element.remove(st)
    spPr = shp._element.spPr
    for e in spPr.findall(qn("a:effectLst")): spPr.remove(e)
    etree.SubElement(spPr, qn("a:effectLst"))
    return shp

def textbox(slide, x, y, w, h, lines, size=9, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, bold_first=False, radius=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE: shp.adjustments[0] = radius
    if fill is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None: shp.line.fill.background()
    else: shp.line.color.rgb = line; shp.line.width = Pt(0.75)
    shp.shadow.inherit = False; noshadow(shp)
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_rich(p, ln, size=size, color=color, align=align, bold=(bold_first and i == 0))
        p.space_after = Pt(0)
    return shp

def arrow(slide, x1, y1, x2, y2, dashed=False, width=1.0, color=LINE, head=True):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    st = c._element.find(qn("p:style"))
    if st is not None: c._element.remove(st)
    c.line.color.rgb = color; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dashed:
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    if head:
        t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c

def curved_arrow(slide, x1, y1, x2, y2, dashed=False, color=LINE, width=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.CURVE, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dashed:
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c

def rect(slide, x, y, w, h, fill, line=None, radius=0.06):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius; s.fill.solid(); s.fill.fore_color.rgb = fill; s.shadow.inherit = False; noshadow(s)
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(0.75)
    return s

# =============================================================== Slide 1: MDP schematic
s = prs.slides.add_slide(blank)
# lanes
for (x, w) in [(0.25, 2.55), (3.1, 3.0), (6.4, 3.35)]:
    rect(s, x, 0.25, w, 4.35, FILL["lane"], radius=0.03)
heads = [("1. Dispatch decision", "subtype unknown; planner uses prior /p/_{σ}", 0.25, 2.55),
         ("2. First hospital", "imaging reveals subtype /σ/", 3.1, 3.0),
         ("3. Stay or transfer", "decided with /σ/ known; treatment times fixed", 6.4, 3.35)]
for (h, sub, x, w) in heads:
    textbox(s, x, 0.3, w, 0.3, [f"*{h}*"], size=10)
    textbox(s, x, 0.55, w, 0.25, [sub], size=8, color=INK2)

# column 1
F = textbox(s, 0.5, 2.15, 2.05, 0.6, ["*Pickup location*", "(ℓ_{0}, /t/_{0}, unknown, ·)"], size=9, fill=FILL["field"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
# column 2
E1 = textbox(s, 3.3, 1.0, 2.6, 0.6, ["*EVT-capable center*", "(ℓ, /t/_{0}+/τ/, known, /σ/)"], size=9, fill=FILL["evt"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
P1 = textbox(s, 3.3, 2.15, 2.6, 0.6, ["*Thrombolysis-capable center*", "(ℓ, /t/_{0}+/τ/, known, /σ/)"], size=9, fill=FILL["psc"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
N1 = textbox(s, 3.3, 3.3, 2.6, 0.6, ["*Non-stroke-center hospital*", "(ℓ, /t/_{0}+/τ/, known, /σ/)"], size=9, fill=FILL["nsc"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
# column 3
E2 = textbox(s, 6.6, 0.9, 2.95, 0.85, ["*Treated at EVT-capable center*", "needle /t/_{a} + /T/_{DTN}", "puncture /t/_{a} + /T/_{DTP} (direct) or /t/_{a} + /T/_{DTP}^{tr} (after transfer)"], size=8, fill=FILL["evt"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
P2 = textbox(s, 6.6, 2.1, 2.95, 0.7, ["*Treated at thrombolysis-capable center*", "needle /t/_{a} + /T/_{DTN}; no EVT on site"], size=8, fill=FILL["psc"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
N2 = textbox(s, 6.6, 3.25, 2.95, 0.7, ["*Remains at non-stroke-center hospital*", "no thrombolysis, no EVT on site; treatment only through a later transfer"], size=8, fill=FILL["nsc"], line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)

# routing arrows (col 1 -> 2)
for (yy) in (1.3, 2.45, 3.6): arrow(s, 2.55, 2.45, 3.3, yy)
textbox(s, 1.35, 2.85, 1.9, 0.3, ["route /a/_{1}: clock + /τ/(ℓ_{0}, ℓ)"], size=8, color=INK2)
# stay arrows (solid)
arrow(s, 5.9, 1.3, 6.6, 1.3); textbox(s, 5.95, 1.05, 0.6, 0.22, ["stay"], size=7.5, color=INK2)
arrow(s, 5.9, 2.45, 6.6, 2.45); textbox(s, 5.95, 2.2, 0.6, 0.22, ["stay"], size=7.5, color=INK2)
arrow(s, 5.9, 3.6, 6.6, 3.6); textbox(s, 5.95, 3.35, 0.6, 0.22, ["stay"], size=7.5, color=INK2)
# transfer arrows (dashed)
arrow(s, 5.9, 2.6, 6.6, 1.62, dashed=True)
arrow(s, 5.9, 3.75, 6.6, 2.72, dashed=True)
arrow(s, 5.9, 3.55, 6.6, 1.72, dashed=True)
textbox(s, 3.3, 4.05, 6.3, 0.3, ["dashed: transfer /a/_{2}, clock + /T/_{DIDO} + /τ/′(ℓ, ℓ′); at most one transfer"], size=8, color=INK2, align=PP_ALIGN.LEFT)
# reward footnote
textbox(s, 0.25, 4.7, 9.5, 0.7, [
    "*Reward* on the terminal state: /P/_{σ}(/t/_{n}, /t/_{p}), the probability of mRS 0–1 at 90 days given onset-to-needle /t/_{n} and onset-to-puncture /t/_{p}.",
    "*First decision* maximizes Σ_{σ} /p/_{σ} [ /R/(/s/, /a/_{1}, /s/′_{σ}) + max_{a₂} /R/(/s/′_{σ}, /a/_{2}, /s/″_{σ}) ];  *second decision* is made with /σ/ known."], size=8, color=INK2, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

# =============================================================== Slide 2: timeline
s = prs.slides.add_slide(blank)
X0, XW = 2.3, 7.2; TMAX = 330.0
def tx(t): return X0 + XW * t / TMAX
lanes = {2: 1.05, 1: 2.05, 0: 3.05}; BH = 0.34
seg_fill = {"pre": RGBColor(0xF0, 0xF0, 0xF0), "travel": RGBColor(0xB9, 0xCD, 0xEE), "dido": RGBColor(0xF6, 0xC9, 0x9F), "wait": RGBColor(0xDA, 0xDA, 0xDA)}
def seg(lane, a, b, kind):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(tx(a)), Inches(lanes[lane] - BH / 2), Inches(tx(b) - tx(a)), Inches(BH))
    r.fill.solid(); r.fill.fore_color.rgb = seg_fill[kind]; r.line.fill.background(); r.shadow.inherit = False; noshadow(r); return r
def tick(lane, t, label, color=INK2, below=False):
    y = lanes[lane]
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx(t)), Inches(y - 0.27), Inches(tx(t)), Inches(y + 0.27)); c.line.color.rgb = color; c.line.width = Pt(1.25)
    textbox(s, tx(t) - 0.5, (y + 0.27) if below else (y - 0.5), 1.0, 0.22, [f"{label} {int(t)}"], size=7.5, color=color)
# axis
ax = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx(0)), Inches(3.75), Inches(tx(330)), Inches(3.75)); ax.line.color.rgb = LINE; ax.line.width = Pt(0.75)
ln = ax.line._get_or_add_ln(); t_ = etree.SubElement(ln, qn("a:tailEnd")); t_.set("type", "triangle")
for t in range(0, 301, 60):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx(t)), Inches(3.72), Inches(tx(t)), Inches(3.8)); c.line.color.rgb = LINE; c.line.width = Pt(0.75)
    textbox(s, tx(t) - 0.3, 3.8, 0.6, 0.22, [str(t)], size=8, color=INK2)
textbox(s, tx(330) + 0.02, 3.62, 0.6, 0.25, ["min"], size=8, color=INK2, align=PP_ALIGN.LEFT)
textbox(s, tx(0) - 0.4, 4.0, 0.8, 0.22, ["onset"], size=8, color=INK2)
# window and pickup lines
w = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx(270)), Inches(0.55), Inches(tx(270)), Inches(3.75)); w.line.color.rgb = RED; w.line.width = Pt(1.0)
d = etree.SubElement(w.line._get_or_add_ln(), qn("a:prstDash")); d.set("val", "dash")
textbox(s, tx(270) - 1.3, 0.3, 2.6, 0.25, ["end of outcome window (270 min)"], size=8, color=RED)
pk = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(tx(60)), Inches(0.55), Inches(tx(60)), Inches(3.75)); pk.line.color.rgb = INK3; pk.line.width = Pt(0.75)
d = etree.SubElement(pk.line._get_or_add_ln(), qn("a:prstDash")); d.set("val", "sysDot")
textbox(s, tx(60) - 0.5, 0.3, 1.0, 0.25, ["pickup"], size=8, color=INK2)
# lane labels
textbox(s, 0.2, lanes[2] - 0.3, 2.0, 0.6, ["*MDP policy*", "direct to EVT-capable center"], size=9, align=PP_ALIGN.RIGHT)
textbox(s, 0.2, lanes[1] - 0.3, 2.0, 0.6, ["*Nearest stroke center*", "thrombolysis-capable, then transfer"], size=9, align=PP_ALIGN.RIGHT)
textbox(s, 0.2, lanes[0] - 0.3, 2.0, 0.6, ["*Nearest hospital*", "non-stroke-center, then transfer"], size=9, align=PP_ALIGN.RIGHT)
# pre-pickup
for L in lanes: seg(L, 0, 60, "pre")
textbox(s, tx(0), lanes[2] - 0.11, tx(60) - tx(0), 0.22, ["onset-to-pickup"], size=7, color=INK3)
# lane 2 (MDP): 25 min drive, DTN 45, DTP 90
seg(2, 60, 85, "travel"); seg(2, 85, 175, "wait")
tick(2, 85, "arrival"); tick(2, 130, "needle", BLUE); tick(2, 175, "puncture", BLUE)
# lane 1 (PSC): 12 min drive, DTN 45, DIDO 121, transfer 25, DTP_tr 60
seg(1, 60, 72, "travel"); seg(1, 72, 193, "dido"); seg(1, 193, 218, "travel"); seg(1, 218, 278, "wait")
tick(1, 72, "arrival"); tick(1, 117, "needle", BLUE); tick(1, 193, "depart"); tick(1, 278, "puncture", RED)
# lane 0 (NSC): 8 min drive, DIDO 121, transfer 25 (to EVT) ; needle via PSC 10 min: 68+121+10+45 = 244
seg(0, 60, 68, "travel"); seg(0, 68, 189, "dido"); seg(0, 189, 214, "travel"); seg(0, 214, 274, "wait")
tick(0, 68, "arrival"); tick(0, 189, "depart"); tick(0, 244, "needle", RED, below=True); tick(0, 274, "puncture", RED)
# legend
lx = 2.3
for (kind, lab) in [("travel", "ambulance travel"), ("dido", "door-in-door-out (/T/_{DIDO} = 121 min)"), ("wait", "in-hospital to treatment"), ("pre", "onset-to-pickup delay")]:
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(lx), Inches(4.45), Inches(0.3), Inches(0.16)); r.fill.solid(); r.fill.fore_color.rgb = seg_fill[kind]; r.line.fill.background(); r.shadow.inherit = False; noshadow(r)
    textbox(s, lx + 0.33, 4.4, 2.2, 0.25, [lab], size=8, color=INK2, align=PP_ALIGN.LEFT); lx += 2.0 if kind != "dido" else 2.45

out = "/mnt/user-data/outputs/schematic/methods_figures_editable.pptx"
prs.save(out); print("saved", out)
