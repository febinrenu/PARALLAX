"""Dataset downloader for P2: resumable, checksummed, manifest per dataset.

    python ml/data/download.py core                 # every dataset needed for training and splits
    python ml/data/download.py isic2018_t3_test fracatlas
    python ml/data/download.py all                  # core + extended (about 7 GB more)
    python ml/data/download.py bdneuro              # human-downloaded set: hashes what you placed
    python ml/data/download.py core --verify        # re-hash files, compare with manifests and lock

Standard library only, so the same file runs on a laptop and inside a Kaggle notebook.
Each dataset gets ml/data/raw/<name>/{manifest.json, files.sha256}. Archive hashes are pinned in
ml/data/checksums.lock.json (committed) so every teammate's download is checked against the same bytes.
Downloaded archives are untrusted: extraction refuses paths that escape the target directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "ml" / "data" / "raw"
DEFAULT_LOCK = REPO_ROOT / "ml" / "data" / "checksums.lock.json"
UA = "parallax-p2-downloader/1.0"
CHUNK = 1 << 20
_ETAG_MD5 = re.compile(r'^"?([0-9a-f]{32})"?$')


class ChecksumError(RuntimeError):
    pass


class UnsafeArchive(RuntimeError):
    pass


class MissingCredentials(RuntimeError):
    """A human step is required (token, accepted rules, manual download)."""


# ----------------------------------------------------------------------------- env and lock


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if v:
            out[k] = v
    return out


def _find_kaggle_credentials() -> bool:
    env = {**read_env_file(REPO_ROOT / ".env"), **os.environ}
    if env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        return True
    home = Path.home() / ".kaggle"
    return (home / "access_token").is_file() or (home / "kaggle.json").is_file()


def _kaggle_env() -> dict[str, str]:
    if not _find_kaggle_credentials():
        raise MissingCredentials(
            "Kaggle credentials not found. Create a token at kaggle.com > Settings > API, then put "
            "KAGGLE_API_TOKEN=... in .env (or ~/.kaggle/access_token). Nothing was downloaded for this dataset."
        )
    return {**os.environ, **{k: v for k, v in read_env_file(REPO_ROOT / ".env").items() if k.startswith("KAGGLE_")}}


class Lock:
    """Pinned archive hashes: the first download of a file writes it, later ones must match."""

    def __init__(self, path: Path = DEFAULT_LOCK):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.is_file():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def check_or_add(self, key: str, sha256: str, nbytes: int) -> None:
        cur = self.entries.get(key)
        if cur is None:
            self.entries[key] = {"sha256": sha256, "bytes": nbytes}
        elif cur["sha256"] != sha256:
            raise ChecksumError(
                f"{key}: sha256 {sha256[:12]}... differs from the pinned {cur['sha256'][:12]}... in {self.path.name}. "
                "The source changed or the file is corrupt; investigate before updating the lock."
            )

    def save(self) -> None:
        """Merge with what is on disk first, so two downloaders running at once do not drop each other's pins."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            for k, v in json.loads(self.path.read_text(encoding="utf-8")).items():
                self.entries.setdefault(k, v)
        self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- http


def _hash_file(path: Path) -> tuple[str, str]:
    s, m = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            s.update(chunk)
            m.update(chunk)
    return s.hexdigest(), m.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_file(path)[0]


def http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_file(
    url: str,
    dest: Path,
    md5: str | None = None,
    retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """Download url to dest with Range resume. Returns {sha256, md5, bytes}. Raises ChecksumError."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        sha, m5 = _hash_file(dest)
        if md5 and m5 != md5.lower():
            raise ChecksumError(f"{dest.name}: existing file has md5 {m5}, expected {md5}")
        return {"sha256": sha, "md5": m5, "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        etag_md5: str | None = None
        total: int | None = None
        try:
            try:
                resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
            except urllib.error.HTTPError as e:
                if e.code == 416 and have:  # the whole file is already in .part
                    resp = None
                else:
                    raise
            if resp is not None:
                with resp:
                    status = getattr(resp, "status", 200)
                    m = _ETAG_MD5.match(resp.headers.get("ETag", ""))
                    etag_md5 = m.group(1) if m else None
                    clen = int(resp.headers.get("Content-Length") or -1)
                    if status == 206:
                        cr = resp.headers.get("Content-Range", "")
                        total = int(cr.rsplit("/", 1)[1]) if "/" in cr and cr.rsplit("/", 1)[1].isdigit() else (have + clen if clen >= 0 else None)
                        mode = "ab"
                    else:
                        total = clen if clen >= 0 else None
                        mode = "wb"  # server ignored Range (or fresh start): restart cleanly
                    t0, shown = time.time(), 0.0
                    with open(part, mode) as f:
                        while chunk := resp.read(CHUNK):
                            f.write(chunk)
                            if time.time() - t0 - shown > 15:
                                shown = time.time() - t0
                                log(f"  {dest.name}: {part.stat().st_size / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else ""))
            size = part.stat().st_size
            if total is not None and size != total:
                raise ConnectionError(f"incomplete: {size} of {total} bytes")
            sha, m5 = _hash_file(part)
            for want, label in ((md5, "expected md5"), (etag_md5, "server ETag")):
                if want and m5 != want.lower():
                    part.unlink(missing_ok=True)
                    raise ChecksumError(f"{dest.name}: md5 {m5} does not match {label} {want}")
            os.replace(part, dest)
            return {"sha256": sha, "md5": m5, "bytes": size}
        except ChecksumError as e:
            last_err = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            last_err = e
        if attempt < retries:
            sleep(min(60, 2**attempt))
    if isinstance(last_err, ChecksumError):
        raise last_err
    raise RuntimeError(f"{url}: giving up after {retries} attempts: {last_err}")


# ----------------------------------------------------------------------------- archives


def safe_extract(archive: Path, out: Path) -> int:
    """Extract a zip, refusing any member that would land outside out. Returns the file count."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for m in members:
            target = (root / m.filename).resolve()
            if root != target and root not in target.parents:
                raise UnsafeArchive(f"{archive.name}: member {m.filename!r} escapes the extraction directory")
        for m in members:
            target = root / m.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    return len(members)


# ----------------------------------------------------------------------------- dataset model


@dataclass
class Item:
    url: str
    filename: str
    extract: bool = False
    md5: str | None = None


@dataclass
class Dataset:
    name: str
    group: str  # core | extended | human
    license: str
    description: str
    items: list[Item] = field(default_factory=list)
    resolve: Callable[[], list[Item]] | None = None  # dynamic items (figshare, dataverse)
    fetch: Callable[["Dataset", Path, "Ctx"], list[dict]] | None = None  # custom (kaggle, ISIC API)
    needs: list[str] = field(default_factory=list)
    source: str = ""
    human_help: str = ""


@dataclass
class Ctx:
    root: Path
    lock: Lock
    sleep: Callable[[float], None] = time.sleep
    keep_archives: bool = False


ISIC = "https://isic-archive.s3.amazonaws.com/challenges/2018/"


def _isic(*names: str, extract: bool | None = None) -> list[Item]:
    return [Item(ISIC + n, n, extract=n.endswith(".zip") if extract is None else extract) for n in names]


def _figshare_items() -> list[Item]:
    files = http_json("https://api.figshare.com/v2/articles/22363012/files")
    return [Item(f["download_url"], f["name"], extract=f["name"].endswith(".zip"), md5=f.get("computed_md5") or f.get("supplied_md5")) for f in files]


def _dataverse_items(doi: str, wanted: list[str]) -> Callable[[], list[Item]]:
    def go() -> list[Item]:
        meta = http_json(f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:{doi}")
        by_name = {f["dataFile"]["filename"]: f["dataFile"] for f in meta["data"]["latestVersion"]["files"]}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise RuntimeError(f"Dataverse doi:{doi} no longer lists {missing}")
        # ?format=original: the default response is a re-encoded .tab whose md5 differs from the API's
        return [Item(f"https://dataverse.harvard.edu/api/access/datafile/{by_name[w]['id']}?format=original", w.replace(".tab", ".csv"), md5=by_name[w].get("md5")) for w in wanted]

    return go


def _kaggle_fetch(kind: str, ref: str, file: str | None = None) -> Callable[[Dataset, Path, Ctx], list[dict]]:
    def go(ds: Dataset, ddir: Path, ctx: Ctx) -> list[dict]:
        env = _kaggle_env()
        exe = shutil.which("kaggle")
        if not exe:
            raise MissingCredentials("The kaggle CLI is not installed. Run: pip install kaggle")
        tmp = ddir / "_dl"
        tmp.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "competitions" if kind == "competition" else "datasets", "download", "-c" if kind == "competition" else "-d", ref, "-p", str(tmp)]
        if file:
            cmd += ["-f", file]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
            if "403" in " ".join(msg) or "rules" in " ".join(msg).lower():
                raise MissingCredentials(f"Kaggle refused {ref}: accept the competition rules in the browser first. ({msg[0]})")
            raise RuntimeError(f"kaggle download {ref} failed: {msg[0]}")
        infos = []
        for f in sorted(tmp.iterdir()):
            sha, m5 = _hash_file(f)
            ctx.lock.check_or_add(f"{ds.name}/{f.name}", sha, f.stat().st_size)
            if f.suffix == ".zip":
                safe_extract(f, ddir)
                if not ctx.keep_archives:
                    f.unlink()
            else:
                shutil.move(str(f), ddir / f.name)
            infos.append({"filename": f.name, "url": f"kaggle:{kind}:{ref}", "bytes": f.stat().st_size if f.exists() else None, "sha256": sha, "md5": m5})
        shutil.rmtree(tmp, ignore_errors=True)
        return infos

    return go


def _isic_meta_fetch(ds: Dataset, ddir: Path, ctx: Ctx) -> list[dict]:
    """Age, sex, site and lesion_id for the official Task 3 test images, from the public ISIC API."""
    gt = next((ctx.root / "isic2018_t3_test").rglob("ISIC2018_Task3_Test_GroundTruth.csv"), None)
    if gt is None:
        raise RuntimeError("isic2018_t3_test must be downloaded first")
    ids = [ln.split(",")[0] for ln in gt.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]

    def one(i: str) -> dict:
        for attempt in range(4):
            try:
                d = http_json(f"https://api.isic-archive.com/api/v2/images/{i}/")
                c = d.get("metadata", {}).get("clinical", {})
                return {"image_id": i, "age": c.get("age_approx"), "sex": c.get("sex"), "site": c.get("anatom_site_general") or c.get("anatom_site_1"),
                        "lesion_id": c.get("lesion_id"), "diagnosis": c.get("diagnosis_3") or c.get("diagnosis_2"), "license": d.get("copyright_license")}
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                ctx.sleep(2**attempt)
        return {"image_id": i, "error": "unreachable"}

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, ids))
    failed = sum(1 for r in rows if "error" in r)
    if failed > len(rows) * 0.02:
        raise RuntimeError(f"ISIC API returned no metadata for {failed} of {len(rows)} images; try again later")
    out = ddir / "isic2018_t3_test_meta.json"
    out.write_text(json.dumps(rows, indent=0), encoding="utf-8")
    sha, m5 = _hash_file(out)
    return [{"filename": out.name, "url": "https://api.isic-archive.com/api/v2/images/", "bytes": out.stat().st_size, "sha256": sha, "md5": m5, "missing": failed}]


def _cifar_items() -> list[Item]:
    return [Item("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz", md5="c58f30108f718f92721af3b95e74349a")]


REGISTRY: dict[str, Dataset] = {}


def _reg(ds: Dataset) -> None:
    REGISTRY[ds.name] = ds


_reg(Dataset("isic2018_t3_train", "core", "CC-BY-NC 4.0", "HAM10000 training images + 7-class labels + lesion groupings (ISIC 2018 Task 3)",
             _isic("ISIC2018_Task3_Training_Input.zip", "ISIC2018_Task3_Training_GroundTruth.zip", "ISIC2018_Task3_Training_LesionGroupings.csv"),
             source="https://challenge.isic-archive.com/data/#2018"))
_reg(Dataset("isic2018_t3_test", "core", "CC-BY-NC 4.0", "Official ISIC 2018 Task 3 test set (1,512 images) + ground truth",
             _isic("ISIC2018_Task3_Test_Input.zip", "ISIC2018_Task3_Test_GroundTruth.zip"), source="https://challenge.isic-archive.com/data/#2018"))
_reg(Dataset("isic2018_t3_test_meta", "core", "CC-BY-NC (per image, from the ISIC API)", "Age, sex, site, lesion_id of the official test images",
             fetch=_isic_meta_fetch, needs=["isic2018_t3_test"], source="https://api.isic-archive.com/"))
_reg(Dataset("ham10000_meta", "core", "CC-BY-NC 4.0", "HAM10000 metadata: dx, age, sex, localization, lesion_id (Harvard Dataverse)",
             resolve=_dataverse_items("10.7910/DVN/DBW86T", ["HAM10000_metadata.tab"]), source="https://doi.org/10.7910/DVN/DBW86T"))
_reg(Dataset("fracatlas", "core", "CC BY 4.0", "FracAtlas: 4,073 musculoskeletal radiographs with COCO/YOLO/VOC annotations",
             resolve=_figshare_items, source="https://doi.org/10.6084/m9.figshare.22363012"))
_reg(Dataset("brain_mri", "core", "CC0 (merged from Br35H, SARTAJ, Figshare; known duplicate/label issues)", "Brain Tumor MRI Dataset, 4 classes, 7,023 images",
             fetch=_kaggle_fetch("dataset", "masoudnickparvar/brain-tumor-mri-dataset"), source="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"))
_reg(Dataset("lgg_seg", "core", "CC BY-NC-SA 4.0 (confirm on dataset page)", "LGG MRI segmentation, 110 TCGA patients",
             fetch=_kaggle_fetch("dataset", "mateuszbuda/lgg-mri-segmentation"), source="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"))
_reg(Dataset("isic2018_t1_eval", "extended", "CC-0", "ISIC 2018 Task 1 validation + test images and masks (skips the 10.4 GB training set)",
             _isic("ISIC2018_Task1-2_Validation_Input.zip", "ISIC2018_Task1_Validation_GroundTruth.zip", "ISIC2018_Task1-2_Test_Input.zip", "ISIC2018_Task1_Test_GroundTruth.zip"),
             source="https://challenge.isic-archive.com/data/#2018"))
_reg(Dataset("rsna_pneumonia", "extended", "Kaggle competition rules (research use)", "RSNA Pneumonia Detection Challenge, DICOM + boxes",
             fetch=_kaggle_fetch("competition", "rsna-pneumonia-detection-challenge"), source="https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge"))
_reg(Dataset("cifar10_test", "extended", "MIT-style (Krizhevsky)", "CIFAR-10, natural-image OOD negatives", _cifar_items(), source="https://www.cs.toronto.edu/~kriz/cifar.html"))
_reg(Dataset("bdneuro", "human", "see Mendeley page (verify)", "BDNeuro-MRI, 5,941 T1-CE images: external brain test",
             source="https://data.mendeley.com/datasets/zwr4ntf94j",
             human_help="Open https://data.mendeley.com/datasets/zwr4ntf94j in a browser, choose Download All, and unzip into ml/data/raw/bdneuro/ (a mendeley.com login may be needed). Then re-run this command."))

GROUPS = {"core": [n for n, d in REGISTRY.items() if d.group == "core"],
          "extended": [n for n, d in REGISTRY.items() if d.group == "extended"]}
GROUPS["all"] = GROUPS["core"] + GROUPS["extended"]


# ----------------------------------------------------------------------------- manifests


def _tree_listing(ddir: Path) -> list[tuple[str, int, str]]:
    skip = {"manifest.json", "files.sha256"}
    rows = []
    for p in sorted(ddir.rglob("*")):
        if p.is_file() and p.name not in skip and p.suffix != ".part" and "_dl" not in p.relative_to(ddir).parts:
            rows.append((p.relative_to(ddir).as_posix(), p.stat().st_size, sha256_file(p)))
    return rows


def _write_manifest(ds: Dataset, ddir: Path, infos: list[dict], how: str) -> dict:
    rows = _tree_listing(ddir)
    (ddir / "files.sha256").write_text("".join(f"{sha}  {size}  {rel}\n" for rel, size, sha in rows), encoding="utf-8")
    m = {
        "name": ds.name, "description": ds.description, "license": ds.license, "source": ds.source,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "how_obtained": how,
        "items": infos, "file_count": len(rows), "total_bytes": sum(r[1] for r in rows),
        "files_sha256_digest": hashlib.sha256((ddir / "files.sha256").read_bytes()).hexdigest(), "complete": True,
    }
    (ddir / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    return m


def register_human(name: str, root: Path) -> dict:
    ds = REGISTRY[name]
    ddir = Path(root) / name
    if not ddir.is_dir() or not any(p.is_file() for p in ddir.rglob("*") if p.name not in ("manifest.json", "files.sha256")):
        raise MissingCredentials(f"{name} is a manual download. {ds.human_help}")
    return _write_manifest(ds, ddir, [], "human")


def run_dataset(ds: Dataset, root: Path, lock: Lock, sleep: Callable[[float], None] = time.sleep, keep_archives: bool = False) -> str:
    root = Path(root)
    ddir = root / ds.name
    if ds.group == "human":
        register_human(ds.name, root)
        return "registered"
    if (ddir / "manifest.json").is_file() and json.loads((ddir / "manifest.json").read_text()).get("complete"):
        return "present"
    ctx = Ctx(root, lock, sleep, keep_archives)
    for dep in ds.needs:
        if not (root / dep / "manifest.json").is_file():
            run_dataset(REGISTRY[dep], root, lock, sleep, keep_archives)
    ddir.mkdir(parents=True, exist_ok=True)
    if ds.fetch is not None:
        infos = ds.fetch(ds, ddir, ctx)
    else:
        items = ds.resolve() if ds.resolve else ds.items
        infos = []
        for it in items:
            print(f"[{ds.name}] {it.filename}", flush=True)
            dest = ddir / it.filename
            info = fetch_file(it.url, dest, md5=it.md5, sleep=sleep)
            lock.check_or_add(f"{ds.name}/{it.filename}", info["sha256"], info["bytes"])
            if it.extract:
                safe_extract(dest, ddir)
                if not keep_archives:
                    dest.unlink()
            infos.append({"filename": it.filename, "url": it.url, **info, "extracted": it.extract})
    _write_manifest(ds, ddir, infos, "script")
    lock.save()
    return "ok"


def run_many(names: list[str], root: Path, lock: Lock, sleep: Callable[[float], None] = time.sleep, keep_archives: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for n in names:
        if n not in REGISTRY:
            results[n] = f"error: unknown dataset (choose from {', '.join(REGISTRY)})"
            continue
        try:
            results[n] = run_dataset(REGISTRY[n], root, lock, sleep, keep_archives)
        except MissingCredentials as e:
            results[n] = f"blocked: {e}"
        except Exception as e:  # keep going: one dead link must not stop the other datasets
            results[n] = f"error: {type(e).__name__}: {e}"
    lock.save()
    return results


def relock_from_manifests(root: Path, lock: Lock) -> int:
    """Add the pinned archive hashes of every manifest on disk to the lock (existing pins must still match)."""
    n = 0
    for mp in sorted(Path(root).glob("*/manifest.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        for it in m.get("items", []):
            if it.get("sha256") and it.get("bytes") and "api.isic-archive.com" not in it.get("url", ""):  # API snapshots can legitimately change
                lock.check_or_add(f"{m['name']}/{it['filename']}", it["sha256"], it["bytes"])
                n += 1
    lock.save()
    return n


def verify_dataset(name: str, root: Path) -> list[str]:
    ddir = Path(root) / name
    if not (ddir / "manifest.json").is_file():
        return [f"{name}: no manifest (not downloaded)"]
    problems = []
    listing = (ddir / "files.sha256").read_text(encoding="utf-8").splitlines()
    for ln in listing:
        sha, size, rel = ln.split("  ", 2)
        p = ddir / rel
        if not p.is_file():
            problems.append(f"{name}/{rel}: missing")
        elif p.stat().st_size != int(size) or sha256_file(p) != sha:
            problems.append(f"{name}/{rel}: content differs from manifest")
    return problems


# ----------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"dataset names or groups: {', '.join(list(GROUPS) + list(REGISTRY))}")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--verify", action="store_true", help="re-hash files against manifests; no downloads")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--relock", action="store_true", help="pin archive hashes from existing manifests into the lock file")
    a = ap.parse_args(argv)
    if a.relock:
        print(f"pinned {relock_from_manifests(a.root, Lock(a.lock))} archive hashes in {a.lock}")
        return 0
    if a.list or not a.names:
        for n, d in REGISTRY.items():
            print(f"{d.group:9s} {n:24s} {d.license:40.40s} {d.description}")
        return 0
    names: list[str] = []
    for n in a.names:
        names += GROUPS.get(n, [n])
    names = list(dict.fromkeys(names))
    if a.verify:
        bad = [p for n in names for p in verify_dataset(n, a.root)]
        print("\n".join(bad) if bad else "all files match their manifests")
        return 1 if bad else 0
    results = run_many(names, a.root, Lock(a.lock), keep_archives=a.keep_archives)
    width = max(len(n) for n in results)
    for n, r in results.items():
        print(f"{n:{width}s}  {r}")
    return 0 if all(r in ("ok", "present", "registered") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
