import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "cv-enrich"
BOARDS = {}


def board(name):
    def deco(fn):
        BOARDS[name] = fn
        return fn
    return deco


def label(b, x, y, text, size=13, fill=INK, anchor="start", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def heat_colour(v):
    lo, hi = (255, 201, 201), (178, 242, 187)
    return "#%02x%02x%02x" % tuple(round(a + (c - a) * v) for a, c in zip(lo, hi))


def heat(b, x, y, w, h, v, text, size=15):
    b.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{heat_colour(max(0, min(1, v)))}" stroke="#adb5bd"/>')
    label(b, x + w / 2, y + h / 2 + size * 0.35, text, size, INK, "middle", "700")


def hbar(b, x, y, width, value, maximum, color, text, name="", name_w=0, h=20):
    c = PALETTE[color]
    if name:
        label(b, x, y + h - 5, name, 13, INK)
    bx = x + name_w
    b.parts.append(f'<rect x="{bx}" y="{y}" width="{width}" height="{h}" rx="5" fill="#e9ecef"/>')
    b.parts.append(f'<rect x="{bx}" y="{y}" width="{max(1.5, width * value / maximum):.1f}" height="{h}" rx="5" fill="{c["stroke"]}"/>')
    label(b, bx + width + 10, y + h - 5, text, 13, c["text"], weight="700")


@board("v1-shape-shortcuts")
def shape_shortcuts():
    b = Board(1120, 520, "Each method has a condition it cannot survive", "3 shapes, 300 training images, 300 test images per column, one seed. Training colours matched the class 90% of the time")
    cols = ["same colours", "colours swapped", "rotated", "noise sd 60"]
    rows = [
        ("colour histogram", [0.883, 0.047, 0.893, 0.637]),
        ("HOG + linear SVM", [0.803, 0.803, 0.550, 0.343]),
        ("tiny CNN", [0.940, 0.013, 0.927, 0.947]),
        ("ResNet18 + logistic", [1.000, 0.990, 0.987, 0.513]),
    ]
    x0, y0, cw, ch = 250, 150, 130, 60
    for j, c in enumerate(cols):
        label(b, x0 + j * (cw + 10) + cw / 2, y0 - 14, c, 12, FAINT, "middle", "700")
    for i, (name, vals) in enumerate(rows):
        y = y0 + i * (ch + 10)
        label(b, 40, y + ch / 2 + 5, name, 14, INK, weight="700")
        for j, v in enumerate(vals):
            heat(b, x0 + j * (cw + 10), y, cw, ch, v, f"{v:.3f}")
    label(b, 40, 460, "Chance is 0.333. A value near zero means the method learned the wrong cue, not that it learned nothing.", 12, FAINT)
    b.card(850, 150, 230, 290, "Read it as", ["colour cue: dies when", "colours are swapped", "HOG: needs upright shapes", "CNN: took the colour", "shortcut, 0.013 swapped", "ResNet: perfect until", "noise it never saw"], "yellow", size=12)
    return b


@board("v1-aliasing-methods")
def aliasing_methods():
    b = Board(1120, 520, "Averaging before sampling is what prevents aliasing", "512 by 512 images reduced to 128 by 128; reference is an ideal low-pass at the new Nyquist limit")
    methods = ["point sample", "bilinear", "bicubic", "area", "gauss 1.0, then point", "gauss 2.0, then point"]
    psnr = [26.62, 29.69, 27.43, 34.37, 32.85, 30.34]
    std = [70.57, 21.96, 36.86, 17.77, 3.48, 0.63]
    label(b, 40, 112, "photograph: PSNR against the reference (dB, higher is better)", 13, FAINT, weight="700")
    for i, (m, v) in enumerate(zip(methods, psnr)):
        hbar(b, 40, 130 + i * 42, 260, v, 40, "green" if v > 32 else "teal", f"{v:.2f}", m, 210)
    label(b, 640, 112, "stripes past the limit: leftover std (lower is better)", 13, FAINT, weight="700")
    for i, (m, v) in enumerate(zip(methods, std)):
        hbar(b, 640, 130 + i * 42, 200, v, 75, "red" if v > 15 else "green", f"{v:.2f}", "", 0)
    b.card(40, 400, 1040, 90, "Surprise", ["area, the method OpenCV recommends for shrinking, still leaves a stripe pattern of std 17.77", "because a box average has side lobes; a Gaussian of sigma 2.0 removes it (0.63) but blurs photographs (30.34)"], "yellow", size=12)
    return b


@board("v1-quantisation-dither")
def quantisation_dither():
    b = Board(1120, 520, "PSNR rewards the banded image, the eye does not", "Smooth ramp quantised to b bits; error measured after a Gaussian of sigma 6 (what the eye integrates)")
    rows = [(5, 40.63, 37.61, 60.64, 63.93), (4, 34.32, 31.33, 41.60, 57.97), (3, 27.70, 24.69, 30.49, 50.68), (2, 20.34, 17.34, 21.44, 43.51)]
    label(b, 40, 110, "bits", 13, FAINT, weight="700")
    label(b, 140, 110, "plain: PSNR / smoothed", 13, FAINT, weight="700")
    label(b, 520, 110, "dithered: PSNR / smoothed", 13, FAINT, weight="700")
    for i, (bits, p, d, ps, ds) in enumerate(rows):
        y = 135 + i * 78
        label(b, 40, y + 28, f"{bits}", 22, INK, weight="700")
        hbar(b, 140, y, 220, p, 70, "orange", f"{p:.2f}", h=18)
        hbar(b, 140, y + 26, 220, ps, 70, "red" if ps < 45 else "green", f"{ps:.2f}", h=18)
        hbar(b, 520, y, 220, d, 70, "orange", f"{d:.2f}", h=18)
        hbar(b, 520, y + 26, 220, ds, 70, "green", f"{ds:.2f}", h=18)
    b.card(880, 140, 210, 300, "Read it as", ["upper bar: PSNR", "lower bar: after", "smoothing", "dither costs about", "3 dB of PSNR", "but lifts the smoothed", "score at 3 bits from", "30.49 to 50.68"], "yellow", size=12)
    return b


@board("v1-colour-space-shift")
def colour_space_shift():
    b = Board(1120, 560, "A colour space helps with one change and not another", "Seven pixel colours, 5-nearest-neighbour classifier, 3,000 test pixels per column")
    cols = ["same light", "shadow", "warm lamp", "strong tint"]
    rows = [
        ("RGB, narrow light", [1.000, 0.676, 1.000, 0.713]),
        ("HSV, narrow light", [1.000, 0.995, 0.995, 0.589]),
        ("HSV hue+sat, narrow", [1.000, 0.997, 0.993, 0.573]),
        ("Lab, narrow light", [1.000, 0.659, 1.000, 0.724]),
        ("Lab a+b, narrow light", [1.000, 0.884, 0.999, 0.581]),
        ("RGB, wide light", [1.000, 0.998, 1.000, 0.703]),
        ("Lab, wide light", [1.000, 0.998, 1.000, 0.620]),
    ]
    x0, y0, cw, ch = 290, 135, 120, 46
    for j, c in enumerate(cols):
        label(b, x0 + j * (cw + 10) + cw / 2, y0 - 12, c, 12, FAINT, "middle", "700")
    for i, (name, vals) in enumerate(rows):
        y = y0 + i * (ch + 8)
        label(b, 40, y + ch / 2 + 5, name, 13, INK, weight="700")
        for j, v in enumerate(vals):
            heat(b, x0 + j * (cw + 10), y, cw, ch, (v - 0.5) / 0.5, f"{v:.3f}", 14)
    b.card(870, 140, 220, 330, "Read it as", ["HSV fixes shadows", "(0.676 to 0.995)", "but not a colour cast", "(0.589 on strong tint)", "training on shadows", "fixes RGB just as well", "(0.998)", "colour scale: 0.5 is red,", "1.0 is green"], "yellow", size=12)
    return b


@board("v1-salt-pepper-filters")
def salt_pepper_filters():
    b = Board(1120, 520, "Median wins on sparse noise and loses on dense noise", "Salt-and-pepper noise on a 512 by 512 photograph; PSNR in dB against the clean image")
    groups = [
        ("5% of pixels", [("mean 3x3", 24.89, "grey"), ("gaussian 5x5", 25.69, "teal"), ("median 3x3", 30.12, "green"), ("median 5x5", 27.85, "orange")]),
        ("20% of pixels", [("mean 3x3", 19.34, "grey"), ("gaussian 5x5", 20.25, "teal"), ("median 3x3", 27.11, "green"), ("median 5x5", 27.25, "orange")]),
        ("40% of pixels", [("mean 3x3", 15.54, "grey"), ("gaussian 5x5", 16.26, "teal"), ("median 3x3", 18.26, "green"), ("median 5x5", 25.32, "orange")]),
    ]
    for g, (title, bars) in enumerate(groups):
        x = 40 + g * 350
        label(b, x, 112, title, 14, INK, weight="700")
        for j, (name, v, colour) in enumerate(bars):
            hbar(b, x, 130 + j * 40, 130, v, 32, colour, f"{v:.2f}", name, 105, h=22)
    b.card(40, 330, 500, 150, "On the clean photograph", ["median 3x3: 30.56 dB, mean 3x3: 29.44 dB", "median 5x5: 28.01 dB", "so the median costs less than the mean when", "there is nothing to remove"], "teal", size=12)
    b.card(580, 330, 500, 150, "On a one-pixel line", ["median 3x3 erases it: brightest value 0", "mean 3x3 keeps a faint ghost: 85", "a median removes thin structure and", "impulses for the same reason"], "red", size=12)
    return b


@board("v1-edge-noise")
def edge_noise():
    b = Board(1120, 520, "Smoothing first matters more than which operator", "Best F1 over thresholds against the true shape boundary, one pixel of tolerance; noise sd in grey levels on a 120-level step")
    sds = [0, 20, 40, 60]
    series = [
        ("Sobel 3x3", [0.999, 0.945, 0.559, 0.333], "orange"),
        ("Scharr", [1.000, 0.937, 0.507, 0.287], "red"),
        ("Sobel 5x5", [0.997, 0.967, 0.880, 0.687], "purple"),
        ("Laplacian", [0.821, 0.158, 0.131, 0.135], "grey"),
        ("Gaussian 1.5 + Sobel 3x3", [0.980, 0.958, 0.890, 0.884], "green"),
        ("Gaussian 1.5 + Laplacian", [0.691, 0.461, 0.338, 0.253], "teal"),
    ]
    x0, x1, y0, y1 = 90, 640, 120, 440
    b.parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="1.5"/>')
    b.parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="1.5"/>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = y1 - t * (y1 - y0)
        label(b, x0 - 10, y + 4, f"{t:.2f}", 11, FAINT, "end")
        b.parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e9ecef"/>')
    px = lambda s: x0 + 30 + s / 60 * (x1 - x0 - 60)
    for s in sds:
        label(b, px(s), y1 + 22, f"sd {s}", 12, FAINT, "middle")
    for name, vals, colour in series:
        c = PALETTE[colour]["stroke"]
        pts = " ".join(f"{px(s):.1f},{y1 - v * (y1 - y0):.1f}" for s, v in zip(sds, vals))
        b.parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3" stroke-linejoin="round"/>')
        for s, v in zip(sds, vals):
            b.parts.append(f'<circle cx="{px(s):.1f}" cy="{y1 - v * (y1 - y0):.1f}" r="4.5" fill="{c}"/>')
    for i, (name, vals, colour) in enumerate(series):
        y = 140 + i * 30
        b.parts.append(f'<rect x="700" y="{y - 11}" width="18" height="12" rx="3" fill="{PALETTE[colour]["stroke"]}"/>')
        label(b, 726, y, name, 13, INK)
        label(b, 1060, y, f"{vals[-1]:.3f}", 13, PALETTE[colour]["text"], "end", "700")
    label(b, 700, 128 - 12, "F1 at noise sd 60", 12, FAINT, weight="700")
    b.card(700, 340, 380, 100, "Read it as", ["Scharr is slightly worse than Sobel in noise", "(0.507 against 0.559 at sd 40)", "a Gaussian before Sobel keeps 0.884 at sd 60"], "yellow", size=12)
    return b


@board("v1-canny-hough-sweeps")
def canny_hough_sweeps():
    b = Board(1120, 540, "Canny has a wide safe window, Hough bins have a narrow one", "Left: Canny F1 at noise sd 25. Right: Hough on four known lines, vote threshold 60, noise 0")
    ratios = [0.1, 0.4, 0.8, 1.0]
    rows = [
        (20, [0.060, 0.060, 0.061, 0.063]),
        (40, [0.062, 0.063, 0.082, 0.102]),
        (80, [0.084, 0.173, 0.513, 0.712]),
        (160, [0.997, 0.997, 0.997, 0.997]),
        (320, [0.997, 0.997, 0.472, 0.143]),
    ]
    x0, y0, cw, ch = 130, 150, 80, 50
    label(b, 40, 118, "high threshold, columns: low/high ratio", 12, FAINT, weight="700")
    for j, r in enumerate(ratios):
        label(b, x0 + j * (cw + 6) + cw / 2, y0 - 8, f"{r}", 12, FAINT, "middle", "700")
    for i, (high, vals) in enumerate(rows):
        y = y0 + i * (ch + 6)
        label(b, 40, y + ch / 2 + 5, f"high {high}", 13, INK, weight="700")
        for j, v in enumerate(vals):
            heat(b, x0 + j * (cw + 6), y, cw, ch, v, f"{v:.3f}", 13)
    cfg = [
        ("rho 0.25, theta 0.25", "0/4", "0/4", "36"),
        ("rho 0.5, theta 0.5", "4/4", "4/4", "63"),
        ("rho 1, theta 1", "4/4", "4/4", "74"),
        ("rho 2, theta 2", "4/4", "4/4", "78"),
        ("rho 4, theta 5", "4/4", "4/4", "79"),
        ("rho 8, theta 10", "0/4", "4/4", "102"),
    ]
    b.table(560, 130, [210, 110, 110, 110], [["bin size", "within 4 px", "within 12 px", "weakest votes"]] + [list(c) for c in cfg], "blue", size=12)
    b.card(560, 400, 520, 110, "Read it as", ["finest bins split each line's votes below 60 and lose it;", "coarsest bins find every line, over 4 px off, and add 2 false ones"], "yellow", size=12)
    return b


def main(names):
    for name in names or list(BOARDS):
        print(BOARDS[name]().save(OUT / f"{name}.svg").relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
