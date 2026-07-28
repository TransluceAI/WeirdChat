"""Single-GPU (Colab A100) override of the qwen3.6-35b-a3b DSpark config.

Same model/geometry as `dspark_qwen36_35b_a3b.py`, sized for one A100 with a
capped baseline corpus:

- global batch 128 at micro-batch 1 (fits 40 GB; raise local_batch_size via
  --opts on an 80 GB card),
- max_length 2048 (halves the target-cache footprint; the WeirdChat
  transcripts are comfortably shorter),
- frequent checkpoints (Colab sessions die — training resumes from
  step_latest automatically),
- torch.compile off (first-run compile time and Colab driver quirks outweigh
  the gain at this scale).

Use exactly like the base config (env vars + --opts from the phase-0 report).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dspark_qwen36_35b_a3b as _base
from dspark_qwen36_35b_a3b import finalize_cfg  # noqa: F401 — collected by load_config

project_name = _base.project_name
exp_name = "dspark_block7_qwen36_35b_a3b_colab"
seed = _base.seed

model = dict(_base.model)

train = dict(
    _base.train,
    local_batch_size=1,
    global_batch_size=128,
    num_train_epochs=4,
    torch_compile=False,
)

logging = dict(
    _base.logging,
    checkpointing_steps=500,
)

data = dict(
    _base.data,
    max_length=2048,
    num_workers=2,
)
