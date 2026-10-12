"""Classical CV stone counter (no ML).

Two independent estimators:
  A. Hough circles (HOUGH_GRADIENT_ALT) on the blurred grayscale image + non-max suppression.
     Primary count. Weakness: any round bright spot (a reflection) looks like a stone.
  B. Outline contours: black-hat morphology picks out the dark girdle ring around each stone,
     Otsu threshold, then count enclosed holes that are round enough. Run at several kernel sizes
     and take the median. Weakness: dark stones and stones without a visible outline.

Agreement between A and B (plus how uniform the detected radii are) sets the confidence.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

WORK_SIZE = 640                # images are resized so their longest side is this
MIN_R_FRAC, MAX_R_FRAC = 0.008, 0.09   # stone radius as a fraction of the short side
HOUGH_PARAM2 = 0.75            # circle "perfectness" for HOUGH_GRADIENT_ALT (0..1)
BLACKHAT_KERNELS = (7, 9, 11, 13)


@dataclass
class CVResult:
    count: int
    confidence: str
    reason: str
    hough_count: int
    contour_count: int
    median_radius_px: float | None
    radius_cv: float | None

    def details(self) -> dict:
        d = asdict(self)
        for key in ("count", "confidence", "reason"):
            d.pop(key)
        return d


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    if abs(scale - 1) < 1e-3:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _nms(circles: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Drop circles whose center lies inside an already accepted (larger) circle."""
    keep: list[tuple[float, float, float]] = []
    for x, y, r in sorted(circles, key=lambda c: (-c[2], c[0], c[1])):
        if any((x - a) ** 2 + (y - b) ** 2 < (0.8 * max(r, c)) ** 2 for a, b, c in keep):
            continue
        keep.append((x, y, r))
    return keep


def hough_circles(gray: np.ndarray) -> list[tuple[float, float, float]]:
    s = min(gray.shape)
    g = cv2.GaussianBlur(gray, (5, 5), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=6, param1=100, param2=HOUGH_PARAM2,
                         minRadius=max(3, int(s * MIN_R_FRAC)), maxRadius=int(s * MAX_R_FRAC))
    if c is None:
        return []
    return _nms([tuple(map(float, x)) for x in c[0]])


def outline_contours(gray: np.ndarray, k: int) -> int:
    s = min(gray.shape)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    min_a, max_a = np.pi * (s * MIN_R_FRAC) ** 2, np.pi * (s * MAX_R_FRAC) ** 2
    n = 0
    for i, c in enumerate(contours):
        if hier[0][i][3] == -1:  # only holes: regions fully enclosed by a dark outline
            continue
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if min_a <= a <= max_a and p > 0 and 4 * np.pi * a / p ** 2 >= 0.6:
            n += 1
    return n


def count_stones(img: np.ndarray) -> CVResult:
    img = _resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = hough_circles(gray)
    hough_n = len(circles)
    contour_n = int(np.median([outline_contours(gray, k) for k in BLACKHAT_KERNELS]))

    radii = np.array([r for _, _, r in circles])
    med_r = round(float(np.median(radii)), 1) if len(radii) else None
    r_cv = round(float(radii.std() / radii.mean()), 3) if len(radii) > 1 else None

    count = hough_n
    diff = abs(hough_n - contour_n)
    rel = diff / max(hough_n, contour_n, 1)
    if hough_n == 0 and contour_n == 0:
        conf, reason = "low", "no stones detected by either method"
    elif diff == 0 or (diff <= max(1, round(0.05 * count)) and min(hough_n, contour_n) > 0):
        conf, reason = "high", f"Hough ({hough_n}) and contour ({contour_n}) methods agree"
    elif rel <= 0.20:
        conf, reason = "medium", f"Hough ({hough_n}) and contour ({contour_n}) differ by {rel:.0%}"
    else:
        conf, reason = "low", f"Hough ({hough_n}) and contour ({contour_n}) disagree by {rel:.0%}"

    if r_cv is not None and r_cv > 0.30 and conf != "low":
        conf = "medium" if conf == "high" else "low"
        reason += f"; detected stone sizes vary a lot (radius CV {r_cv})"

    return CVResult(count=count, confidence=conf, reason=reason, hough_count=hough_n,
                    contour_count=contour_n, median_radius_px=med_r, radius_cv=r_cv)
