"""Build Kaggle notebook directly from committed joint trainer source."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
folder = root / "kaggle"


def cell(code):
    return {"cell_type": "code", "execution_count": None,
            "id": hashlib.sha256(code.encode()).hexdigest()[:8],
            "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}


trainer = (root / "training/train_joint.py").read_text()
trainer = trainer.split('if __name__ == "__main__":')[0]
notebook = {
    "cells": [
        cell('!pip -q install "transformers>=4.48,<5" "datasets>=3,<5" "accelerate>=1,<2"\n'),
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
             'import torch\nassert torch.cuda.is_available(), "No GPU allocated"\n'
             'print("GPU:", torch.cuda.get_device_name(0))\n'),
        cell(trainer),
        cell('main("/kaggle/working/verse-joint-v1", train_samples=6000, '
             'eval_samples=1000, epochs=1, batch_size=8)\n'),
    ],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                  "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
folder.mkdir(exist_ok=True)
(folder / "verse-lawyer-joint.ipynb").write_text(json.dumps(notebook, indent=1))
meta = {
    "id": "ffatty/verse-lawyer-modernbert-149m-nli-baseline",
    "title": "Verse Lawyer ModernBERT 149M Joint",
    "code_file": "verse-lawyer-joint.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [],
    "kernel_sources": [], "model_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("Built joint-training notebook", len(notebook["cells"]), "cells")
