"""The server's dataset store (on the persistent volume at LATTICE_DATASET_DIR).

Each dataset lives in <root>/<dataset_id>/:
  meta.json                   name, classes, counts, created_at, sha256 of the uploaded zip
  train_images.npy, train_labels.npy
                              uncompressed, so they can be memory-mapped: a slice reads
                              only its own rows instead of loading the whole set into RAM
  test.npz                    the held-out split, for server-side evaluation

Uploads are processed in the background (start_build): the HTTP request returns at once,
and the website polls upload_status() for progress. Builds run one at a time, reading the
zip from disk, so a large upload can't exhaust the server's memory or time out a proxy.
"""
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from shared.data import slice_data
from shared.datasets import IMAGE_SIZE, pack_arrays, unpack_arrays
from shared.protocol import DATA_SEED

log = logging.getLogger("lattice.datasets")

MAX_ZIP_BYTES = 200 * 1024 * 1024            # size of the uploaded file
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024  # guards against zip bombs
MAX_FILES = 60_000
MIN_PER_CLASS = 5
TEST_FRACTION = 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
UPLOAD_STATUS_TTL = 3600                     # forget finished upload statuses after an hour
Image.MAX_IMAGE_PIXELS = 40_000_000          # refuse absurdly large images


class DatasetError(ValueError):
    """The upload can't be turned into a dataset; the message says why, in plain words."""


def _to_square_pixels(data: bytes) -> np.ndarray:
    """Any image -> 32x32 RGB uint8: centre-crop to a square, then resize."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "dataset"
    return f"{s}-{uuid.uuid4().hex[:6]}"


def _sha256(source) -> str:
    h = hashlib.sha256()
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    else:
        h.update(source)
    return h.hexdigest()


class DatasetStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._train = {}                        # dataset_id -> (images memmap, labels)
        self._slices = OrderedDict()            # (id, slice_id, num_slices) -> packed bytes
        self._max_cached_slices = 12
        self._uploads = {}                      # upload_id -> status dict
        self._builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-build")

    # ------------------------------------------------------------ reading
    def _dir(self, dataset_id: str) -> Path:
        if not ID_PATTERN.match(dataset_id or ""):
            raise DatasetError("invalid dataset id")
        path = self.root / dataset_id
        if not (path / "meta.json").is_file():
            raise DatasetError(f"no dataset '{dataset_id}'")
        return path

    def meta(self, dataset_id: str) -> dict:
        return json.loads((self._dir(dataset_id) / "meta.json").read_text())

    def list(self) -> list:
        metas = []
        for d in self.root.iterdir():
            if (d / "meta.json").is_file():
                metas.append(json.loads((d / "meta.json").read_text()))
        return sorted(metas, key=lambda m: m["created_at"])

    def _train_arrays(self, dataset_id: str):
        with self._lock:
            if dataset_id not in self._train:
                d = self._dir(dataset_id)
                self._train[dataset_id] = (np.load(d / "train_images.npy", mmap_mode="r"),
                                           np.load(d / "train_labels.npy"))
            return self._train[dataset_id]

    def train_size(self, dataset_id: str) -> int:
        return len(self._train_arrays(dataset_id)[1])

    def slice_bytes(self, dataset_id: str, slice_id: int, num_slices: int) -> bytes:
        """The packed .npz of one volunteer's slice; same slicing as shared.data.slice_data."""
        key = (dataset_id, slice_id, num_slices)
        with self._lock:
            if key in self._slices:
                self._slices.move_to_end(key)
                return self._slices[key]
        images, labels = self._train_arrays(dataset_id)
        idx = np.sort(np.asarray(slice_data(len(labels), slice_id, num_slices), dtype=np.int64))
        data = pack_arrays(np.asarray(images[idx]), labels[idx])
        with self._lock:
            self._slices[key] = data
            while len(self._slices) > self._max_cached_slices:
                self._slices.popitem(last=False)
        return data

    def test_arrays(self, dataset_id: str):
        return unpack_arrays((self._dir(dataset_id) / "test.npz").read_bytes())

    def delete(self, dataset_id: str):
        """Remove a dataset's folder and forget anything cached from it."""
        folder = self._dir(dataset_id)              # validates the id; raises if it doesn't exist
        with self._lock:
            self._train.pop(dataset_id, None)
            for key in [k for k in self._slices if k[0] == dataset_id]:
                del self._slices[key]
        shutil.rmtree(folder)

    # ------------------------------------------------------------ writing
    def add_arrays(self, name, classes, train_images, train_labels, test_images, test_labels, source, extra=None):
        dataset_id = _slug(name)
        d = self.root / dataset_id
        d.mkdir(parents=True)
        np.save(d / "train_images.npy", np.ascontiguousarray(train_images, dtype=np.uint8))
        np.save(d / "train_labels.npy", np.asarray(train_labels, dtype=np.uint16))
        (d / "test.npz").write_bytes(pack_arrays(test_images, test_labels))
        meta = {"id": dataset_id, "name": name, "classes": list(classes), "num_classes": len(classes),
                "train_count": int(len(train_labels)), "test_count": int(len(test_labels)),
                "image_size": IMAGE_SIZE, "source": source, "created_at": time.time(), **(extra or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def build_from_zip(self, name: str, source, progress=None) -> dict:
        """A zip of class folders (a file path, or bytes) -> a stored dataset.
        progress(done, total) is called as images are prepared. Raises DatasetError with a reason."""
        size = os.path.getsize(source) if isinstance(source, (str, Path)) else len(source)
        if size > MAX_ZIP_BYTES:
            raise DatasetError(f"the zip is larger than {MAX_ZIP_BYTES // (1024 * 1024)} MB")

        # The same file again (e.g. a retry after a timeout) returns the dataset it already became.
        digest = _sha256(source)
        for existing in self.list():
            if existing.get("sha256") == digest:
                return {**existing, "already_existed": True}

        try:
            zf = zipfile.ZipFile(source if isinstance(source, (str, Path)) else io.BytesIO(source))
        except zipfile.BadZipFile:
            raise DatasetError("that file isn't a valid zip")
        with zf:
            entries = [i for i in zf.infolist() if not i.is_dir()
                       and Path(i.filename).suffix.lower() in IMAGE_EXTENSIONS
                       and not any(part.startswith((".", "__MACOSX")) for part in Path(i.filename).parts)]
            if len(entries) > MAX_FILES:
                raise DatasetError(f"too many images (the limit is {MAX_FILES:,})")
            if sum(i.file_size for i in entries) > MAX_UNCOMPRESSED_BYTES:
                raise DatasetError("the images are too large in total once unzipped")

            # The class is the folder directly containing the image; a single shared top folder is ignored.
            by_class, skipped = {}, 0
            for n, info in enumerate(entries, 1):
                parts = Path(info.filename).parts
                if len(parts) < 2:
                    skipped += 1                       # an image not inside any class folder
                else:
                    try:
                        by_class.setdefault(parts[-2], []).append(_to_square_pixels(zf.read(info)))
                    except Exception:
                        skipped += 1                   # unreadable or not really an image
                if progress and (n % 50 == 0 or n == len(entries)):
                    progress(n, len(entries))

        classes = sorted(c for c, imgs in by_class.items() if len(imgs) >= MIN_PER_CLASS)
        if len(classes) < 2:
            raise DatasetError(f"need at least 2 class folders with {MIN_PER_CLASS}+ images each "
                               f"(found: {', '.join(f'{c} ({len(v)})' for c, v in by_class.items()) or 'none'})")

        # Stratified split: TEST_FRACTION of every class (at least one image) goes to the test set.
        rng = np.random.default_rng(DATA_SEED)
        tr_x, tr_y, te_x, te_y = [], [], [], []
        for label, c in enumerate(classes):
            imgs = np.stack(by_class[c])
            order = rng.permutation(len(imgs))
            n_test = max(1, int(round(len(imgs) * TEST_FRACTION)))
            te_x.append(imgs[order[:n_test]]); te_y += [label] * n_test
            tr_x.append(imgs[order[n_test:]]); tr_y += [label] * (len(imgs) - n_test)
        meta = self.add_arrays(name, classes, np.concatenate(tr_x), np.asarray(tr_y),
                               np.concatenate(te_x), np.asarray(te_y), source="upload", extra={"sha256": digest})
        meta["skipped_files"] = skipped
        return meta

    # ------------------------------------------------------------ background uploads
    def start_build(self, name: str, path: Path) -> dict:
        """Queue a zip (already saved at path) for building; returns its status at once.
        The file is deleted when the build finishes, whatever the outcome."""
        upload_id = uuid.uuid4().hex[:12]
        status = {"upload_id": upload_id, "name": name, "state": "queued", "processed": 0, "total": None,
                  "meta": None, "error": None, "updated_at": time.time()}
        with self._lock:
            self._forget_old_uploads()
            self._uploads[upload_id] = status
        self._builder.submit(self._run_build, upload_id, name, Path(path))
        return dict(status)

    def upload_status(self, upload_id: str):
        with self._lock:
            status = self._uploads.get(upload_id)
            return dict(status) if status else None

    def _update(self, upload_id: str, **changes):
        with self._lock:
            self._uploads[upload_id].update(changes, updated_at=time.time())

    def _run_build(self, upload_id: str, name: str, path: Path):
        self._update(upload_id, state="processing")
        try:
            meta = self.build_from_zip(name, path, progress=lambda done, total:
                                       self._update(upload_id, processed=done, total=total))
            self._update(upload_id, state="done", meta=meta)
        except DatasetError as e:
            self._update(upload_id, state="error", error=str(e))
        except Exception:
            log.exception("building dataset '%s' failed", name)
            self._update(upload_id, state="error", error="the upload couldn't be processed")
        finally:
            path.unlink(missing_ok=True)

    def _forget_old_uploads(self):
        now = time.time()
        for uid in [u for u, s in self._uploads.items()
                    if s["state"] in ("done", "error") and now - s["updated_at"] > UPLOAD_STATUS_TTL]:
            del self._uploads[uid]