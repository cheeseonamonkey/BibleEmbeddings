"""Build reproducible private Kaggle continuation with embedded small datasets."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "kaggle-domain"
out.mkdir(exist_ok=True)

def code_cell(source: str) -> dict:
    return {"cell_type":"code","execution_count":None,
            "id":hashlib.sha256(source.encode()).hexdigest()[:8],
            "metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}

def payload(path: Path):
    raw=path.read_bytes()
    return base64.b64encode(zlib.compress(raw,level=9)).decode(),hashlib.sha256(raw).hexdigest()

domain, domain_hash=payload(root/"training/domain_pairs.jsonl")
eval_data, eval_hash=payload(root/"evaluation/seed_pairs.jsonl")
trainer=(root/"training/train_domain.py").read_text().split('if __name__ == "__main__":')[0]

setup=(
    'import os\nos.environ["CUDA_VISIBLE_DEVICES"]="0"\n'
    'import base64, zlib, hashlib\nfrom pathlib import Path\n'
    'Path("/kaggle/working/datasets").mkdir(exist_ok=True)\n'
    f'DOMAIN_B64={domain!r}\nEVAL_B64={eval_data!r}\n'
    f'DOMAIN_SHA={domain_hash!r}\nEVAL_SHA={eval_hash!r}\n'
    'for name,blob,digest in [("domain",DOMAIN_B64,DOMAIN_SHA),("dev",EVAL_B64,EVAL_SHA)]:\n'
    '    raw=zlib.decompress(base64.b64decode(blob))\n'
    '    assert hashlib.sha256(raw).hexdigest()==digest, f"{name} corrupted"\n'
    '    Path(f"/kaggle/working/datasets/{name}.jsonl").write_bytes(raw)\n'
    '    print(name,len(raw),"sha256",digest)\n'
)
preflight=(
    'import torch\nfrom pathlib import Path\n'
    'assert torch.cuda.is_available(), "GPU not allocated"\n'
    'print("GPU",torch.cuda.get_device_name(0))\n'
    'matches = sorted(Path("/kaggle/input").rglob("verse-joint-v1/model/config.json"))\n'
    'matches = [p.parent for p in matches if list(p.parent.glob("*.safetensors"))]\n'
    'assert len(matches)==1, f"Expected one mounted v1 checkpoint, found: {matches}"\n'
    'CHECKPOINT=str(matches[0])\n'
    'print("Continuing v1",CHECKPOINT)\n'
)
invoke=(
    'main(checkpoint=CHECKPOINT, '
    'domain_jsonl="/kaggle/working/datasets/domain.jsonl", '
    'bible_eval_jsonl="/kaggle/working/datasets/dev.jsonl", '
    'output="/kaggle/working/verse-domain-v1", '
    'domain_repeats=4, general_samples=2200, epochs=2, batch_size=8)\n'
)
notebook={
    "cells":[
        code_cell('!pip -q install "transformers>=4.48,<5" "datasets>=3,<5" "accelerate>=1,<2"\n'),
        code_cell(setup),
        code_cell(preflight),
        code_cell(trainer),
        code_cell(invoke),
    ],
    "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
                "language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5,
}
(out/"verse-lawyer-domain.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
metadata={
    "id":"ffatty/verse-lawyer-domain-v1-149m",
    "title":"Verse Lawyer Domain V1 149M",
    "code_file":"verse-lawyer-domain.ipynb",
    "language":"python","kernel_type":"notebook",
    "is_private":True,"enable_gpu":True,"enable_internet":True,
    "dataset_sources":[],
    "kernel_sources":["ffatty/verse-lawyer-modernbert-149m-joint"],
    "model_sources":[],
}
(out/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print("Generated",out,"cells",len(notebook["cells"]),
      "data",domain_hash[:12],"dev",eval_hash[:12])
