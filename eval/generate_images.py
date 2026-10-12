"""Generate synthetic top-down jewelry photos with a known stone count.

Each scene: lit background -> metal piece (ring band, pendant disc or bracelet strip) -> round stones
with a dark girdle, radial shading and a specular highlight. Difficulty knobs: blur, sensor noise,
lighting gradient, stone overlap, stone size, dark (colored) stones, low stone/metal contrast.

Run:  python eval/generate_images.py      (deterministic: fixed seeds)
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "images"
SIZE = 640

METALS = {  # BGR
    "yellow_gold": (60, 165, 212),
    "white_gold": (196, 198, 200),
    "rose_gold": (130, 150, 205),
    "silver": (205, 205, 210),
    "platinum": (185, 186, 188),
}
STONES = {  # BGR body color
    "diamond": (245, 245, 248),
    "cubic_zirconia": (250, 248, 245),
    "sapphire": (150, 55, 20),
    "ruby": (45, 25, 175),
    "emerald": (70, 140, 30),
}


def background(rng, base: int, gradient: float) -> np.ndarray:
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    g = base + gradient * (x / SIZE - 0.5) * 2 * 60
    img = np.dstack([g, g, g * 0.97]) + rng.normal(0, 3, (SIZE, SIZE, 3))
    return np.clip(img, 0, 255).astype(np.float32)


def shade(img, mask, color, strength=0.25):
    """Fill the mask with a color and soft top-left to bottom-right shading."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    s = 1.0 + strength * (0.5 - (x + y) / (2 * SIZE))
    for c in range(3):
        ch = img[..., c]
        ch[mask > 0] = (color[c] * s)[mask > 0]


def draw_stone(img, cx, cy, r, color):
    yy, xx = np.ogrid[0:SIZE, 0:SIZE]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = d <= r
    # radial body shading: brighter center, darker edge
    falloff = np.clip(1.0 - 0.45 * (d / max(r, 1)) ** 2, 0, 1)
    for c in range(3):
        ch = img[..., c]
        ch[inside] = color[c] * falloff[inside]
    # dark girdle outline
    girdle = (d <= r) & (d >= r - max(1.5, r * 0.14))
    img[girdle] = img[girdle] * 0.45
    # specular highlight
    hr = max(1, int(r * 0.28))
    cv2.circle(img, (int(cx - r * 0.3), int(cy - r * 0.3)), hr, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def ring_positions(n, cx, cy, radius, start=0.0):
    return [(cx + radius * math.cos(start + 2 * math.pi * i / n), cy + radius * math.sin(start + 2 * math.pi * i / n)) for i in range(n)]


def grid_positions(n, cx, cy, pitch, cols):
    rows = math.ceil(n / cols)
    pts = []
    for i in range(n):
        r_, c_ = divmod(i, cols)
        offset = pitch / 2 if r_ % 2 else 0
        pts.append((cx + (c_ - (cols - 1) / 2) * pitch + offset, cy + (r_ - (rows - 1) / 2) * pitch))
    return pts


def render(spec: dict) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(spec["seed"])
    img = background(rng, spec.get("bg", 70), spec.get("gradient", 0.2))
    metal = METALS[spec["metal"]]
    stone = STONES[spec["stone"]]
    cx = cy = SIZE / 2
    layout = spec["layout"]
    r = spec["stone_r"]
    mask = np.zeros((SIZE, SIZE), np.uint8)

    if layout == "halo":  # ring seen from above: band + circle of stones around a center
        cv2.circle(mask, (int(cx), int(cy)), 230, 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), 150, 0, -1)
        pts = ring_positions(spec["n"], cx, cy, 190, rng.uniform(0, 1))
    elif layout == "pave":  # pendant disc covered in a staggered grid
        cv2.circle(mask, (int(cx), int(cy)), 250, 255, -1)
        pts = grid_positions(spec["n"], cx, cy, spec["pitch"], spec["cols"])
    elif layout == "line":  # bracelet strip, one or two rows
        cv2.rectangle(mask, (20, int(cy - 70)), (SIZE - 20, int(cy + 70)), 255, -1)
        rows = spec.get("rows", 1)
        per_row = math.ceil(spec["n"] / rows)
        pts = []
        for row in range(rows):
            yrow = cy + (row - (rows - 1) / 2) * spec["pitch"]
            k = min(per_row, spec["n"] - len(pts))
            pts += [(45 + (SIZE - 90) * (i + 0.5) / per_row, yrow) for i in range(k)]
    elif layout == "cluster":  # center stone + petals
        cv2.circle(mask, (int(cx), int(cy)), 220, 255, -1)
        pts = [(cx, cy)] + ring_positions(spec["n"] - 1, cx, cy, spec["pitch"], 0.3)
    elif layout == "solitaire":
        cv2.circle(mask, (int(cx), int(cy)), 230, 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), 170, 0, -1)
        pts = [(cx, cy - 200)]
    else:
        raise ValueError(layout)

    shade(img, mask, metal)
    for (px, py) in pts:
        jitter_r = r * rng.uniform(1 - spec.get("r_jitter", 0.05), 1 + spec.get("r_jitter", 0.05))
        draw_stone(img, px + rng.normal(0, spec.get("pos_jitter", 0.5)), py + rng.normal(0, spec.get("pos_jitter", 0.5)), jitter_r, stone)

    for _ in range(spec.get("reflections", 0)):  # round specular glints on the metal: stone look-alikes
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(160, 220) if layout == "halo" else rng.uniform(60, 200)
        gx, gy = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
        cv2.circle(img, (int(gx), int(gy)), int(r * rng.uniform(0.7, 1.0)), (250, 250, 250), -1, lineType=cv2.LINE_AA)

    if spec.get("light"):  # strong uneven lighting: one side much darker
        y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        img *= (0.35 + 0.75 * (x / SIZE))[..., None]
    if spec.get("blur"):
        k = spec["blur"] | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if spec.get("noise"):
        img += rng.normal(0, spec["noise"], img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if spec.get("downscale"):  # low-resolution phone shot, upscaled back
        small = cv2.resize(out, (SIZE // spec["downscale"], SIZE // spec["downscale"]), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if spec.get("jpeg"):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, spec["jpeg"]])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out, len(pts)


SCENES = [
    {"file": "img01_halo12_clean.png", "seed": 1, "layout": "halo", "n": 12, "stone_r": 18, "metal": "white_gold", "stone": "diamond", "bg": 60,
     "difficulty": "clean"},
    {"file": "img02_pave20_rose.png", "seed": 2, "layout": "pave", "n": 20, "stone_r": 22, "pitch": 56, "cols": 5, "metal": "rose_gold", "stone": "diamond",
     "noise": 6, "difficulty": "mild noise"},
    {"file": "img03_channel8_sapphire.png", "seed": 3, "layout": "line", "n": 8, "stone_r": 24, "pitch": 0, "metal": "white_gold", "stone": "sapphire",
     "difficulty": "dark colored stones"},
    {"file": "img04_halo16_blur_noise.png", "seed": 4, "layout": "halo", "n": 16, "stone_r": 16, "metal": "white_gold", "stone": "diamond",
     "blur": 9, "noise": 14, "difficulty": "heavy blur + noise"},
    {"file": "img05_pave30_small_platinum.png", "seed": 5, "layout": "pave", "n": 30, "stone_r": 11, "pitch": 30, "cols": 6, "metal": "platinum", "stone": "diamond",
     "difficulty": "small stones, low metal/stone contrast"},
    {"file": "img06_halo24_touching.png", "seed": 6, "layout": "halo", "n": 24, "stone_r": 26, "metal": "yellow_gold", "stone": "diamond",
     "difficulty": "stones touching/overlapping"},
    {"file": "img07_halo14_cz_lighting.png", "seed": 7, "layout": "halo", "n": 14, "stone_r": 17, "metal": "silver", "stone": "cubic_zirconia",
     "light": True, "difficulty": "strong uneven lighting"},
    {"file": "img08_cluster7_ruby.png", "seed": 8, "layout": "cluster", "n": 7, "stone_r": 30, "pitch": 70, "metal": "yellow_gold", "stone": "ruby",
     "difficulty": "colored stones, cluster"},
    {"file": "img09_tennis40_two_rows.png", "seed": 9, "layout": "line", "n": 40, "stone_r": 12, "rows": 2, "pitch": 30, "metal": "yellow_gold", "stone": "diamond",
     "difficulty": "many small stones"},
    {"file": "img10_solitaire1.png", "seed": 10, "layout": "solitaire", "n": 1, "stone_r": 38, "metal": "yellow_gold", "stone": "diamond",
     "difficulty": "single large stone"},
    {"file": "img11_halo22_overlap_jpeg.png", "seed": 11, "layout": "halo", "n": 22, "stone_r": 31, "r_jitter": 0.15, "pos_jitter": 3, "metal": "platinum",
     "stone": "diamond", "jpeg": 25, "difficulty": "heavy overlap + JPEG artifacts"},
    {"file": "img12_pave24_lowres.png", "seed": 12, "layout": "pave", "n": 24, "stone_r": 10, "pitch": 26, "cols": 6, "metal": "silver",
     "stone": "cubic_zirconia", "downscale": 5, "difficulty": "low resolution (128px source)"},
    {"file": "img13_halo12_reflections.png", "seed": 13, "layout": "halo", "n": 12, "stone_r": 16, "metal": "white_gold", "stone": "diamond",
     "reflections": 5, "difficulty": "round reflections on metal (false positives)"},
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    for scene in SCENES:
        img, n = render(scene)
        cv2.imwrite(str(OUT / scene["file"]), img)
        labels[scene["file"]] = {"true_count": n, "difficulty": scene["difficulty"], "metal": scene["metal"], "stone": scene["stone"]}
        print(f"{scene['file']:<36} true_count={n:<3} {scene['difficulty']}")
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2))


if __name__ == "__main__":
    main()
