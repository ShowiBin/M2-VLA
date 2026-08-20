M2-VLA
================

This repository contains the implementation of paper [M2-VLA:  Boosting Vision-Language Models for Generalizable Manipulation via Layer Mixture and Meta-Skills](https://arxiv.org/abs/2604.24182)

Python Environment
------------------
```bash
conda create -n m2-vla python=3.10 -y
conda activate m2-vla

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

LIBERO Setup
------------

Install LIBERO in the same conda environment and add it to `PYTHONPATH`.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .

cd /path/to/M2-VLA
python -m pip install -r experiments/robot/libero/libero_requirements.txt

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/LIBERO
```

Checkpoint Layout
-----------------

Put released checkpoints under `CKPT/` using the following names:

| Suite | `--task_suite_name` | Checkpoint directory |
| --- | --- | --- |
| Spatial | `libero_spatial` | `CKPT/M2-VLA-spatial` |
| Object | `libero_object` | `CKPT/M2-VLA-object` |
| Goal | `libero_goal` | `CKPT/M2-VLA-goal` |
| Long | `libero_10` | `CKPT/M2-VLA-long` |

Checkpoint Download
-------------------

The released checkpoints are available from Hugging Face:

- Repository: https://huggingface.co/GmanGmanGman/M2-VLA
- Spatial: https://huggingface.co/GmanGmanGman/M2-VLA/tree/main/M2VLA-spatial
- Object: https://huggingface.co/GmanGmanGman/M2-VLA/tree/main/M2VLA-object
- Goal: https://huggingface.co/GmanGmanGman/M2-VLA/tree/main/M2VLA-goal
- Long: https://huggingface.co/GmanGmanGman/M2-VLA/tree/main/M2VLA-long

To download all four checkpoints with the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
mkdir -p CKPT

hf download GmanGmanGman/M2-VLA \
  --include "M2VLA-spatial/*" \
  --include "M2VLA-object/*" \
  --include "M2VLA-goal/*" \
  --include "M2VLA-long/*" \
  --local-dir CKPT

mv CKPT/M2VLA-spatial CKPT/M2-VLA-spatial
mv CKPT/M2VLA-object CKPT/M2-VLA-object
mv CKPT/M2VLA-goal CKPT/M2-VLA-goal
mv CKPT/M2VLA-long CKPT/M2-VLA-long
```


Evaluation
----------

Example:

```bash
conda activate m2-vla
cd /path/to/M2-VLA

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/LIBERO

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint CKPT/M2-VLA-goal \
  --task_suite_name libero_goal \
  --use_pro_version True
```

Use the checkpoint table above to switch between LIBERO suites.

Acknowledgment
--------------------
We thank VLA-Adapter for their open-sourced work!

- VLA-Adapter: [https://github.com/OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter)
- openVLA: [https://github.com/OpenHelix-Team/VLA-Adapter](https://github.com/openvla/openvla)
