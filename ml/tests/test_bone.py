from __future__ import annotations

import numpy as np
import pandas as pd

from ml.train import bone


def test_read_yolo_boxes_converts_center_format(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.2 0.2\n")
    b = bone.read_yolo_boxes(p)
    assert np.allclose(b[0], [0.4, 0.3, 0.6, 0.7]) and np.allclose(b[1], [0.0, 0.0, 0.2, 0.2])
    (tmp_path / "e.txt").write_text("")
    assert bone.read_yolo_boxes(tmp_path / "e.txt").shape == (0, 4)


def test_build_yolo_dataset_keeps_negatives_with_empty_labels(tmp_path):
    src = tmp_path / "raw" / "fracatlas" / "FracAtlas"
    (src / "images" / "Fractured").mkdir(parents=True)
    (src / "images" / "Non_fractured").mkdir(parents=True)
    (src / "Annotations" / "YOLO").mkdir(parents=True)
    rows = []
    for i, (sub, split, boxes) in enumerate([("Fractured", "train", "0 0.5 0.5 0.1 0.1\n"), ("Non_fractured", "train", ""), ("Non_fractured", "test", ""), ("Fractured", "dropped", "0 0.5 0.5 0.1 0.1\n")]):
        (src / "images" / sub / f"IMG{i}.jpg").write_bytes(b"x")
        (src / "Annotations" / "YOLO" / f"IMG{i}.txt").write_text(boxes)
        rows.append({"source_dir": "fracatlas/FracAtlas", "relpath": f"images/{sub}/IMG{i}.jpg", "image_id": f"IMG{i}", "split": split, "yolo_label": f"Annotations/YOLO/IMG{i}.txt"})
    yaml = bone.build_yolo_dataset(pd.DataFrame(rows), tmp_path / "raw", tmp_path / "yolo")
    assert (tmp_path / "yolo" / "labels" / "train" / "IMG1.txt").read_text() == ""
    assert (tmp_path / "yolo" / "images" / "test" / "IMG2.jpg").exists()
    assert not (tmp_path / "yolo" / "images" / "dropped").exists()
    assert "names:\n  0: fracture" in yaml.read_text() and "test: images/test" in yaml.read_text()


def test_pack_unpack_roundtrip():
    dets = [np.array([[0, 0, 1, 1, 0.5]]), np.zeros((0, 5)), np.array([[0, 0, 0.5, 0.5, 0.9], [0.1, 0.1, 0.2, 0.2, 0.3]])]
    packed = bone.pack_dets(dets)
    back = bone.unpack_dets(packed["det_rows"], packed["det_offsets"])
    assert [len(d) for d in back] == [1, 0, 2] and np.allclose(back[2], dets[2])


def _split(n, rng, signal):
    dets, gts = [], []
    for i in range(n):
        pos = rng.rand() < 0.3
        g = np.array([[0.2, 0.2, 0.4, 0.4]]) if pos else np.zeros((0, 4))
        gts.append(g)
        if pos and rng.rand() < 0.85:
            dets.append(np.array([[0.2, 0.2, 0.4, 0.4, rng.uniform(0.4, 1.0) * signal]]))
        elif rng.rand() < 0.2:
            dets.append(np.array([[0.6, 0.6, 0.8, 0.8, rng.uniform(0.001, 0.4)]]))
        else:
            dets.append(np.zeros((0, 5)))
    return {"dets": dets, "gts": gts, "groups": np.arange(n)}


def test_evaluate_reports_cis_and_uses_validation_threshold():
    rng = np.random.RandomState(0)
    val, test = _split(400, rng, 1.0), _split(400, rng, 1.0)
    m = bone.evaluate(val, test, B=200)
    for k in ("map50", "auroc_image", "sensitivity_at_val_90spec", "specificity_at_val_threshold"):
        assert m[k]["lo"] <= m[k]["point"] <= m[k]["hi"], k
    assert m["auroc_image"]["point"] > 0.85 and m["map50"]["point"] > 0.6
    assert 0.8 < m["specificity_at_val_threshold"]["point"] <= 1.0
    assert m["n_fractured"] == sum(len(g) > 0 for g in test["gts"])


def test_read_image_bgr_decodes_truncated_jpeg(tmp_path):
    import io

    from PIL import Image

    arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    p = tmp_path / "t.jpg"
    p.write_bytes(buf.getvalue()[:-30])  # lose the tail, like the FracAtlas files
    with np.testing.assert_raises(OSError):
        Image.open(p).load()
    img = bone.read_image_bgr(p)
    assert img.shape == (64, 64, 3)


def test_predict_split_passes_arrays_not_paths(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        Image.fromarray(np.full((20, 30, 3), 50 * i, np.uint8)).save(tmp_path / f"{i}.jpg")
        paths.append(tmp_path / f"{i}.jpg")
    seen = []

    class _T:
        def cpu(self):
            return self

        def numpy(self):
            return self.a

        def __init__(self, a):
            self.a = a

    class _Boxes:
        def __init__(self):
            self.xyxyn, self.conf = _T(np.array([[0.1, 0.1, 0.2, 0.2]])), _T(np.array([0.5]))

        def __len__(self):
            return 1

    class _R:
        boxes = _Boxes()

    class _M:
        def predict(self, src, **kw):
            seen.append(type(src[0]))
            return [_R() for _ in src]

    out = bone.predict_split(_M(), paths, batch=2)
    assert len(out) == 3 and out[0].shape == (1, 5) and all(t is np.ndarray for t in seen)
