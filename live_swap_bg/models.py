"""Download the two models. Both are Apache-2.0; they are fetched from their publishers, not shipped here."""

import urllib.request
from pathlib import Path

from .matte import VITMATTE_DIR
from .seed import SAM2_CHECKPOINT

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/" + SAM2_CHECKPOINT
VITMATTE_REPO = "hustvl/" + VITMATTE_DIR


def download_models(models: Path) -> dict:
    from huggingface_hub import snapshot_download

    models.mkdir(parents=True, exist_ok=True)
    checkpoint = models / SAM2_CHECKPOINT
    if not checkpoint.is_file():
        partial = checkpoint.with_suffix(".part")
        print(f"downloading {SAM2_URL} (~900 MB)", flush=True)
        urllib.request.urlretrieve(SAM2_URL, partial)
        partial.rename(checkpoint)
    snapshot_download(VITMATTE_REPO, local_dir=models / VITMATTE_DIR,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    return {"sam2": str(checkpoint), "vitmatte": str(models / VITMATTE_DIR)}
