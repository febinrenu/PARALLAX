"""Decode PNG / JPG / DICOM into a display copy and an analysis copy.

display  uint8, windowed, what a person (and the viewer) sees. HxW, or HxWx3 for colour.
analysis float32 in [0, 1], full dynamic range, no windowing. HxW, or HxWx3 for colour.

For DICOM the modality LUT (rescale slope / intercept) is applied first, then the display
copy is windowed from the file's window tags (default: 0.5 to 99.5 percentile). MONOCHROME1
is inverted on both copies so that bone is always bright and air is always dark.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_BYTES = 256 * 1024 * 1024
MAX_PIXELS = 100_000_000
DEFAULT_WINDOW_PCT = (0.5, 99.5)
_DICOM_MAGIC_OFFSET = 128


class DecodeError(ValueError):
    """The input is not a decodable image. The message is safe to show to a user."""


@dataclass
class DecodedImage:
    display: np.ndarray
    analysis: np.ndarray
    sha256: str
    source_format: str  # "dicom" | "png" | "jpeg" | other PIL format name, lowercase
    is_color: bool = False
    photometric: str | None = None
    pixel_spacing: tuple[float, float] | None = None
    bits_stored: int | None = None
    window: tuple[float, float] | None = None  # (center, width) in modality units
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.display.shape[0]), int(self.display.shape[1])

    def meta(self) -> dict:
        """JSON-safe description with no pixel data and no patient tags."""
        return {
            "sha256": self.sha256,
            "format": self.source_format,
            "height": self.shape[0],
            "width": self.shape[1],
            "is_color": self.is_color,
            "photometric": self.photometric,
            "pixel_spacing": list(self.pixel_spacing) if self.pixel_spacing else None,
            "bits_stored": self.bits_stored,
            "window": list(self.window) if self.window else None,
        }


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_dicom(raw: bytes) -> bool:
    return raw[_DICOM_MAGIC_OFFSET : _DICOM_MAGIC_OFFSET + 4] == b"DICM"


def _to_bytes(source: bytes | bytearray | str | Path) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    path = Path(source)
    if path.stat().st_size > MAX_BYTES:
        raise DecodeError("File is too large to process.")
    return path.read_bytes()


def load_image(source: bytes | bytearray | str | Path) -> DecodedImage:
    """Decode an upload. Raises DecodeError with a user-safe message on any failure."""
    try:
        raw = _to_bytes(source)
    except OSError as exc:
        raise DecodeError("The file could not be read.") from exc
    if not raw:
        raise DecodeError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise DecodeError("File is too large to process.")
    digest = sha256_hex(raw)
    if is_dicom(raw):
        return _decode_dicom(raw, digest)
    return _decode_raster(raw, digest)


def _window_from_percentiles(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, DEFAULT_WINDOW_PCT)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    return float((lo + hi) / 2.0), float(hi - lo)


def _apply_window(x: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    out = (x.astype(np.float32) - lo) / max(width, 1e-6)
    return np.clip(out, 0.0, 1.0)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x.astype(np.float32) - lo) / (hi - lo)).astype(np.float32)


def _first(value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def _decode_dicom(raw: bytes, digest: str) -> DecodedImage:
    import pydicom

    warnings: list[str] = []
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        arr = ds.pixel_array
    except Exception as exc:  # pydicom raises many unrelated types on bad or unsupported files
        raise DecodeError(
            "The DICOM file could not be decoded (corrupt, or an unsupported compression)."
        ) from exc

    samples = int(getattr(ds, "SamplesPerPixel", 1))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frames > 1:
        arr = arr[0]
        warnings.append(f"multi-frame DICOM ({frames} frames): using frame 0")
    if samples == 3:
        return _decode_dicom_color(ds, arr, digest, warnings)
    if arr.ndim != 2:
        raise DecodeError("Unsupported DICOM pixel layout.")
    if arr.size > MAX_PIXELS:
        raise DecodeError("Image is too large to process.")

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    x = arr.astype(np.float32)
    slope = _first(getattr(ds, "RescaleSlope", None)) or 1.0
    intercept = _first(getattr(ds, "RescaleIntercept", None)) or 0.0
    x = x * np.float32(slope) + np.float32(intercept)

    center = _first(getattr(ds, "WindowCenter", None))
    width = _first(getattr(ds, "WindowWidth", None))
    if center is None or width is None or width <= 0:
        center, width = _window_from_percentiles(x)
        warnings.append("no window tags: used 0.5 to 99.5 percentile window")

    display_f = _apply_window(x, center, width)
    analysis = _minmax(x)
    if photometric == "MONOCHROME1":
        display_f = 1.0 - display_f
        analysis = 1.0 - analysis
    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    pixel_spacing = None
    if spacing is not None and len(spacing) == 2:
        pixel_spacing = (float(spacing[0]), float(spacing[1]))
    return DecodedImage(
        display=np.rint(display_f * 255.0).astype(np.uint8),
        analysis=analysis.astype(np.float32),
        sha256=digest,
        source_format="dicom",
        photometric=photometric,
        pixel_spacing=pixel_spacing,
        bits_stored=int(getattr(ds, "BitsStored", 0)) or None,
        window=(float(center), float(width)),
        warnings=warnings,
    )


def _decode_dicom_color(ds, arr: np.ndarray, digest: str, warnings: list[str]) -> DecodedImage:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise DecodeError("Unsupported DICOM colour layout.")
    display = np.clip(arr, 0, 255).astype(np.uint8)
    return DecodedImage(
        display=display,
        analysis=(display.astype(np.float32) / 255.0),
        sha256=digest,
        source_format="dicom",
        is_color=True,
        photometric=str(getattr(ds, "PhotometricInterpretation", "RGB")),
        bits_stored=int(getattr(ds, "BitsStored", 0)) or None,
        warnings=warnings,
    )


def _decode_raster(raw: bytes, digest: str) -> DecodedImage:
    warnings: list[str] = []
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "unknown").lower()
            if im.width * im.height > MAX_PIXELS:
                raise DecodeError("Image is too large to process.")
            im = ImageOps.exif_transpose(im)
            im.load()
            mode = im.mode
            if mode in ("I;16", "I;16L", "I;16B", "I"):
                a = np.asarray(im).astype(np.float32)
                bits = 16
                lo, hi = a.min(), a.max()
                center, width = _window_from_percentiles(a)
                display = np.rint(_apply_window(a, center, width) * 255.0).astype(np.uint8)
                analysis = _minmax(a)
                return DecodedImage(
                    display=display,
                    analysis=analysis,
                    sha256=digest,
                    source_format=fmt,
                    photometric="MONOCHROME2",
                    bits_stored=bits,
                    window=(center, width),
                    warnings=warnings,
                )
            if mode in ("RGBA", "LA", "P", "PA", "CMYK", "1"):
                im = im.convert("RGB" if mode not in ("LA", "1") else "L")
                mode = im.mode
            if mode == "L":
                a8 = np.asarray(im, dtype=np.uint8)
                return DecodedImage(
                    display=a8,
                    analysis=a8.astype(np.float32) / 255.0,
                    sha256=digest,
                    source_format=fmt,
                    photometric="MONOCHROME2",
                    bits_stored=8,
                    warnings=warnings,
                )
            if mode == "RGB":
                a8 = np.asarray(im, dtype=np.uint8)
                if np.array_equal(a8[..., 0], a8[..., 1]) and np.array_equal(a8[..., 1], a8[..., 2]):
                    g = a8[..., 0]
                    warnings.append("RGB file with identical channels: treated as grayscale")
                    return DecodedImage(
                        display=g,
                        analysis=g.astype(np.float32) / 255.0,
                        sha256=digest,
                        source_format=fmt,
                        photometric="MONOCHROME2",
                        bits_stored=8,
                        warnings=warnings,
                    )
                return DecodedImage(
                    display=a8,
                    analysis=a8.astype(np.float32) / 255.0,
                    sha256=digest,
                    source_format=fmt,
                    is_color=True,
                    photometric="RGB",
                    bits_stored=8,
                    warnings=warnings,
                )
            raise DecodeError(f"Unsupported image mode: {mode}.")
    except DecodeError:
        raise
    except Exception as exc:  # PIL raises many unrelated types on corrupt input
        raise DecodeError("The file is not a readable PNG, JPEG or DICOM image.") from exc
