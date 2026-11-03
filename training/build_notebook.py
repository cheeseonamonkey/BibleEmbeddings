"""Generate Kaggle notebook from the exact committed trainer source."""
import json
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "kaggle"
folder.mkdir(parents=True, exist_ok=True)


def cell(source):
    return {
        "cell_type": "code", "execution_count": None, "id": hashlib.sha256(source.encode()).hexdigest()[:8],
        "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)
    }


trainer = (root / "training/train_nli.py").read_text()
nb = {
    "cells": [
        cell('!pip -q install "transformers>=4.48,<5" "datasets>=3,<5" "accelerate>=1,<2" scikit-learn\n'),
        cell('import torch\nassert torch.cuda.is_available(), "No GPU allocated"\nprint("GPU:", torch.cuda.get_device_name(0))\n'),
        cell(trainer.split('if __name__ == "__main__":')[0] + '\n'),
        cell('main("/kaggle/working/verse-nli-v0", train_samples=6000, eval_samples=1000, epochs=1.0)\n'),
    ],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(folder / "verse-lawyer-nli.ipynb").write_text(json.dumps(nb, indent=1))
metadata = {
    "id": "ffatty/verse-lawyer-modernbert-149m-nli-baseline",
    "title": "Verse Lawyer — ModernBERT 149M NLI Baseline",
    "code_file": "verse-lawyer-nli.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
print("Generated", folder / "verse-lawyer-nli.ipynb")
