import os
import sys
import tempfile
from datetime import datetime

cache_root = os.path.join(tempfile.gettempdir(), "jp_quant_cache")
mpl_cache_root = os.path.join(cache_root, "matplotlib")
os.makedirs(mpl_cache_root, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", cache_root)
os.environ.setdefault("MPLCONFIGDIR", mpl_cache_root)

import matplotlib

from config import LOG_ROOT, MODEL_VERSION


matplotlib.use("Agg")


class TeeWriter:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, "a", buffering=1)
        self.original = original_stream

    def write(self, message):
        self.original.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def isatty(self):
        return False


_img_counter = [0]


def create_run_dirs(run_label=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_name = f"{MODEL_VERSION}_{run_label}_{timestamp}" if run_label else f"{MODEL_VERSION}_{timestamp}"
    run_dir = os.path.join(LOG_ROOT, run_name)
    img_dir = os.path.join(run_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "img_dir": img_dir,
        "log_file": os.path.join(run_dir, "log.out"),
        "err_file": os.path.join(run_dir, "log.err"),
    }


def setup_logging(run_paths):
    sys.stdout = TeeWriter(run_paths["log_file"], sys.__stdout__)
    sys.stderr = TeeWriter(run_paths["err_file"], sys.__stderr__)
    print("=" * 60)
    print(f"  Run:  {run_paths['run_name']}")
    print(f"  Dir:  {run_paths['run_dir']}/")
    print("  Out:  log.out | Err: log.err | Img: img/")
    print("=" * 60)
    print()


def save_fig(plt_module, img_dir, name=None):
    _img_counter[0] += 1
    fig = plt_module.gcf()
    if name is None:
        title = fig._suptitle.get_text() if fig._suptitle else None
        if not title:
            for ax in fig.axes:
                if ax.get_title():
                    title = ax.get_title()
                    break
        name = title or f"figure_{_img_counter[0]:02d}"
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
    safe = safe.strip().replace(" ", "_")[:80]
    path = os.path.join(img_dir, f"{_img_counter[0]:02d}_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[LOG] Saved: {path}")
    return path


def get_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to select a device.") from exc
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
