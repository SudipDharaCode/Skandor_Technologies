import json

import cv2
import numpy as np

from app.vision import count_stones, decode_image


def test_counts_clean_synthetic_image_exactly(images_dir):
    r = count_stones(cv2.imread(str(images_dir / "img01_halo12_clean.png")))
    assert r.count == 12
    assert r.confidence == "high"


def test_cv_is_deterministic(images_dir):
    img = cv2.imread(str(images_dir / "img05_pave30_small_platinum.png"))
    assert count_stones(img) == count_stones(img)


def test_hard_image_gets_lower_confidence(images_dir):
    # Round reflections on the metal fool Hough; the outline method disagrees -> not high.
    r = count_stones(cv2.imread(str(images_dir / "img13_halo12_reflections.png")))
    assert r.confidence != "high"


def test_blank_image_is_low_confidence():
    r = count_stones(np.full((300, 300, 3), 128, np.uint8))
    assert r.count == 0 and r.confidence == "low"


def test_decode_rejects_garbage_bytes():
    assert decode_image(b"definitely not an image") is None


def test_labels_file_matches_images(images_dir):
    labels = json.loads((images_dir / "labels.json").read_text())
    assert len(labels) >= 5
    for f in labels:
        assert (images_dir / f).exists()
