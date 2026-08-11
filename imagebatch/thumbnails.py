"""Cached thumbnails.

Serving a few hundred full-resolution images to a browser grid is slow, so the
gallery renders cached JPEG thumbnails instead. The cache lives outside the
album directories so it never shows up as album content.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)


class ThumbnailCache:
    def __init__(self, cache_dir: Path, size: int = 384) -> None:
        self.cache_dir = Path(cache_dir)
        self.size = size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, source: Path) -> Path:
        try:
            stat = source.stat()
            fingerprint = f"{source}|{stat.st_mtime_ns}|{stat.st_size}|{self.size}"
        except OSError:
            fingerprint = f"{source}|{self.size}"
        digest = hashlib.blake2b(fingerprint.encode("utf-8"), digest_size=12).hexdigest()
        return self.cache_dir / f"{digest}.jpg"

    def get(self, source: Path) -> str:
        """Return a thumbnail path, falling back to the original on any error."""
        source = Path(source)
        cached = self._cache_path(source)
        if cached.is_file():
            return str(cached)
        try:
            with Image.open(source) as img:
                img = img.convert("RGB")
                img.thumbnail((self.size, self.size), Image.LANCZOS)
                img.save(cached, format="JPEG", quality=82)
        except (OSError, UnidentifiedImageError) as exc:
            log.debug("Thumbnail failed for %s (%s); serving original", source, exc)
            return str(source)
        return str(cached)

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*.jpg"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed
