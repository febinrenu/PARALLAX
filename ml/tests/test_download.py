from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from ml.data import download as dl


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_plain_download_hashes_and_size(server, tmp_path):
    data = b"x" * 5000
    server.files["/a.bin"] = data
    dest = tmp_path / "a.bin"
    info = dl.fetch_file(server.base + "/a.bin", dest, sleep=lambda s: None)
    assert dest.read_bytes() == data
    assert info["sha256"] == hashlib.sha256(data).hexdigest()
    assert info["bytes"] == 5000
    assert not (tmp_path / "a.bin.part").exists()


def test_resume_after_truncated_part(server, tmp_path):
    data = bytes(range(256)) * 40
    server.files["/r.bin"] = data
    dest = tmp_path / "r.bin"
    (tmp_path / "r.bin.part").write_bytes(data[:3000])
    info = dl.fetch_file(server.base + "/r.bin", dest, sleep=lambda s: None)
    assert dest.read_bytes() == data
    assert info["sha256"] == hashlib.sha256(data).hexdigest()
    assert server.hits[0][1] == "bytes=3000-"


def test_resume_survives_dropped_connection(server, tmp_path):
    data = b"abcdefgh" * 2000
    server.files["/d.bin"] = data
    server.fail_after["/d.bin"] = 4000
    dest = tmp_path / "d.bin"
    dl.fetch_file(server.base + "/d.bin", dest, sleep=lambda s: None)
    assert dest.read_bytes() == data
    assert len(server.hits) >= 2


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    data = b"0123456789" * 500
    server.files["/i.bin"] = data
    server.ignore_range = True
    dest = tmp_path / "i.bin"
    (tmp_path / "i.bin.part").write_bytes(data[:100])
    dl.fetch_file(server.base + "/i.bin", dest, sleep=lambda s: None)
    assert dest.read_bytes() == data


def test_md5_mismatch_is_rejected_and_not_kept(server, tmp_path):
    server.files["/m.bin"] = b"hello"
    dest = tmp_path / "m.bin"
    with pytest.raises(dl.ChecksumError):
        dl.fetch_file(server.base + "/m.bin", dest, md5="0" * 32, retries=1, sleep=lambda s: None)
    assert not dest.exists()


def test_s3_style_etag_md5_is_checked_but_multipart_etag_is_ignored(server, tmp_path):
    data = b"payload" * 100
    server.files["/e.bin"] = data
    server.etags["/e.bin"] = hashlib.md5(data).hexdigest()
    dl.fetch_file(server.base + "/e.bin", tmp_path / "e.bin", sleep=lambda s: None)
    server.files["/mp.bin"] = data
    server.etags["/mp.bin"] = "abc123-9"  # multipart etag is not an md5
    dl.fetch_file(server.base + "/mp.bin", tmp_path / "mp.bin", sleep=lambda s: None)
    server.files["/bad.bin"] = data
    server.etags["/bad.bin"] = "f" * 32
    with pytest.raises(dl.ChecksumError):
        dl.fetch_file(server.base + "/bad.bin", tmp_path / "bad.bin", retries=1, sleep=lambda s: None)


def test_lockfile_pins_first_hash_and_rejects_drift(tmp_path):
    lock = dl.Lock(tmp_path / "lock.json")
    lock.check_or_add("ds/a.zip", "aa" * 32, 10)
    lock.save()
    lock2 = dl.Lock(tmp_path / "lock.json")
    lock2.check_or_add("ds/a.zip", "aa" * 32, 10)  # same: fine
    with pytest.raises(dl.ChecksumError):
        lock2.check_or_add("ds/a.zip", "bb" * 32, 10)


def test_safe_extract_blocks_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    z.write_bytes(_zip({"../escape.txt": b"nope", "ok.txt": b"fine"}))
    out = tmp_path / "out"
    with pytest.raises(dl.UnsafeArchive):
        dl.safe_extract(z, out)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_counts_files(tmp_path):
    z = tmp_path / "ok.zip"
    z.write_bytes(_zip({"d/a.txt": b"1", "d/b.txt": b"2"}))
    n = dl.safe_extract(z, tmp_path / "o")
    assert n == 2 and (tmp_path / "o" / "d" / "b.txt").read_bytes() == b"2"


def test_dataset_flow_writes_manifest_lock_and_is_idempotent(server, tmp_path):
    zbytes = _zip({"img/1.jpg": b"aaa", "img/2.jpg": b"bbb"})
    server.files["/in.zip"] = zbytes
    server.files["/gt.csv"] = b"image,label\n1,a\n"
    ds = dl.Dataset(
        name="toy",
        group="core",
        license="CC0",
        description="toy",
        items=[
            dl.Item(url=server.base + "/in.zip", filename="in.zip", extract=True),
            dl.Item(url=server.base + "/gt.csv", filename="gt.csv"),
        ],
    )
    lock = dl.Lock(tmp_path / "lock.json")
    dl.run_dataset(ds, tmp_path / "raw", lock, sleep=lambda s: None)
    m = json.loads((tmp_path / "raw" / "toy" / "manifest.json").read_text())
    assert m["license"] == "CC0" and m["how_obtained"] == "script"
    assert {i["filename"] for i in m["items"]} == {"in.zip", "gt.csv"}
    assert m["file_count"] == 3  # two images + csv
    assert (tmp_path / "raw" / "toy" / "img" / "1.jpg").exists()
    assert not (tmp_path / "raw" / "toy" / "in.zip").exists()  # archive dropped after extraction
    listing = (tmp_path / "raw" / "toy" / "files.sha256").read_text().splitlines()
    assert len(listing) == 3
    assert "toy/in.zip" in json.loads((tmp_path / "lock.json").read_text())
    hits = len(server.hits)
    dl.run_dataset(ds, tmp_path / "raw", lock, sleep=lambda s: None)
    assert len(server.hits) == hits, "second run must not re-download"


def test_verify_detects_a_modified_file(server, tmp_path):
    server.files["/gt.csv"] = b"image,label\n1,a\n"
    ds = dl.Dataset("toy2", "core", "CC0", "t", [dl.Item(url=server.base + "/gt.csv", filename="gt.csv")])
    lock = dl.Lock(tmp_path / "lock.json")
    dl.run_dataset(ds, tmp_path / "raw", lock, sleep=lambda s: None)
    assert dl.verify_dataset("toy2", tmp_path / "raw") == []
    (tmp_path / "raw" / "toy2" / "gt.csv").write_bytes(b"tampered")
    problems = dl.verify_dataset("toy2", tmp_path / "raw")
    assert problems and "gt.csv" in problems[0]


def test_kaggle_dataset_without_token_fails_clearly_and_others_continue(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(dl, "_find_kaggle_credentials", lambda: False)
    ds = dl.REGISTRY["brain_mri"]
    with pytest.raises(dl.MissingCredentials) as e:
        dl.run_dataset(ds, tmp_path / "raw", dl.Lock(tmp_path / "l.json"))
    assert "KAGGLE_API_TOKEN" in str(e.value)
    results = dl.run_many(["brain_mri", "nonexistent"], tmp_path / "raw", dl.Lock(tmp_path / "l.json"))
    assert results["brain_mri"].startswith("blocked")
    assert results["nonexistent"].startswith("error")


def test_human_dataset_verify_registers_placed_files(tmp_path):
    d = tmp_path / "raw" / "bdneuro"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"1")
    (d / "sub").mkdir()
    (d / "sub" / "b.jpg").write_bytes(b"22")
    dl.register_human("bdneuro", tmp_path / "raw")
    m = json.loads((d / "manifest.json").read_text())
    assert m["how_obtained"] == "human" and m["file_count"] == 2


def test_human_dataset_missing_folder_gives_instructions(tmp_path):
    with pytest.raises(dl.MissingCredentials) as e:
        dl.register_human("bdneuro", tmp_path / "raw")
    assert "mendeley" in str(e.value).lower()


def test_env_file_parser_does_not_override_and_ignores_comments(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nKAGGLE_API_TOKEN=abc\nEMPTY=\nQ='quoted'\n")
    monkeypatch.setenv("KEEP", "1")
    env = dl.read_env_file(f)
    assert env == {"KAGGLE_API_TOKEN": "abc", "Q": "quoted"}
