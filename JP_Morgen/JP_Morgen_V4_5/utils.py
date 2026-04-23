import os
import sys
import tempfile
import json
import random
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


def _slugify(text):
    safe_chars = []
    for char in str(text):
        if char.isalnum():
            safe_chars.append(char.lower())
        elif char in (" ", "-", "_", ".", "="):
            safe_chars.append("-")
    slug = "".join(safe_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _write_run_manifest(run_paths, manifest=None, header_lines=None):
    if manifest is not None:
        with open(run_paths["manifest_json"], "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
    if header_lines is not None:
        with open(run_paths["manifest_txt"], "w", encoding="utf-8") as file:
            file.write("\n".join(header_lines).rstrip() + "\n")


def create_run_dirs(run_label=None, experiment_name=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    experiment_slug = _slugify(experiment_name) if experiment_name else ""
    run_label_slug = _slugify(run_label) if run_label else ""
    if run_label_slug:
        name_slug = run_label_slug
    elif experiment_slug:
        name_slug = experiment_slug
    else:
        name_slug = ""

    if name_slug:
        run_name = "__".join([MODEL_VERSION, name_slug, timestamp])
    else:
        run_name = "__".join([MODEL_VERSION, timestamp])
    run_dir = os.path.join(LOG_ROOT, run_name)
    img_dir = os.path.join(run_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "img_dir": img_dir,
        "log_file": os.path.join(run_dir, f"{run_name}.out"),
        "err_file": os.path.join(run_dir, f"{run_name}.err"),
        "manifest_json": os.path.join(run_dir, "run_manifest.json"),
        "manifest_txt": os.path.join(run_dir, "run_manifest.txt"),
        "run_label": run_label,
        "experiment_name": experiment_name,
    }


def setup_logging(run_paths, header_lines=None, manifest=None):
    sys.stdout = TeeWriter(run_paths["log_file"], sys.__stdout__)
    sys.stderr = TeeWriter(run_paths["err_file"], sys.__stderr__)
    _write_run_manifest(run_paths, manifest=manifest, header_lines=header_lines)
    print("=" * 60)
    print(f"  Run:  {run_paths['run_name']}")
    print(f"  Dir:  {run_paths['run_dir']}/")
    print(f"  Out:  {os.path.basename(run_paths['log_file'])}")
    print(f"  Err:  {os.path.basename(run_paths['err_file'])}")
    print(f"  Img:  {os.path.basename(run_paths['img_dir'])}/")
    print(f"  Meta: {os.path.basename(run_paths['manifest_json'])} | {os.path.basename(run_paths['manifest_txt'])}")
    print("=" * 60)
    if header_lines:
        for line in header_lines:
            print(line)
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


def set_random_seed(seed, deterministic=False):
    if seed is None:
        return None

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to set the project seed.") from exc

    np.random.seed(seed)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to set the project seed.") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    return seed
