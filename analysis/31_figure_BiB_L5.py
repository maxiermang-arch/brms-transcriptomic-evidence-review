"""Combine Figure6 (survival) and Figure7 (PDO) into a single BiB Figure 6.

Stacked vertically at 174 mm width, 300 dpi. Panel labels A/B added.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

MM = 1 / 25.4
DPI = 300
W_INCH = 174 * MM
W_PX = int(W_INCH * DPI)
GAP_PX = 24
LABEL_SIZE = 22  # px, ~6.6 pt at 300 dpi

FIGDIR = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'figures'))
OUT = FIGDIR / "Figure6_L5_survival_pdo"


def load(path):
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


surv = load(FIGDIR / "Figure6_survival_cohortB.png")
pdo = load(FIGDIR / "Figure7_pdo_ic50.png")

# scale both to W_PX
h_surv = int(surv.height * W_PX / surv.width)
h_pdo = int(pdo.height * W_PX / pdo.width)
surv = surv.resize((W_PX, h_surv), Image.LANCZOS)
pdo = pdo.resize((W_PX, h_pdo), Image.LANCZOS)

# create white canvas
H = h_surv + GAP_PX + h_pdo
canvas = Image.new("RGBA", (W_PX, H), (255, 255, 255, 255))
canvas.paste(surv, (0, 0), surv)
canvas.paste(pdo, (0, h_surv + GAP_PX), pdo)

# flatten (white background)
final = Image.new("RGB", (W_PX, H), (255, 255, 255))
final.paste(canvas, mask=canvas.split()[3])

draw = ImageDraw.Draw(final)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", LABEL_SIZE)
except Exception:
    font = ImageFont.load_default()

# panel labels
draw.text((10, 6), "A", fill="black", font=font)
draw.text((10, h_surv + GAP_PX + 6), "B", fill="black", font=font)

final.save(f"{OUT}.png", dpi=(DPI, DPI))
final.save(f"{OUT}.pdf", resolution=DPI)
print("OK Figure6_L5_survival_pdo")
