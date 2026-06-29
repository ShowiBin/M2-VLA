M2-VLA
================

This repository contains the released code and LIBERO evaluation workflow for M2-VLA.
It is intended to be a compact release package: model weights, generated
rollouts, and local experiment logs are not included.

Repository Layout
-----------------

- `experiments/robot/libero/run_libero_eval.py`: LIBERO evaluation entry point.
- `experiments/robot/libero/libero_utils.py`: LIBERO environment, image, video,
  and quaternion utilities.
- `experiments/robot/openvla_utils.py`: checkpoint loading and policy component
  utilities.
- `experiments/robot/robot_utils.py`: policy wrapper utilities.
- `prismatic/`: model definitions and Hugging Face classes required by the
  released checkpoints.
- `CKPT/`: local checkpoint directory. This directory is ignored by Git.
- `eval_logs/`, `experiments/logs/`, `rollouts/`: runtime output directories.

Python Environment
------------------

The setup below follows the dependency layout used by VLA-Adapter and pins the
core runtime versions used by this release package.

```bash
conda create -n m2-vla python=3.10 -y
conda activate m2-vla

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

If your PyTorch/CUDA stack has already been configured on the machine, install
the matching PyTorch build first and then install the remaining requirements.
The default `requirements.txt` uses `torch==2.2.0`, `torchvision==0.17.0`, and
`torchaudio==2.2.0`.

Some checkpoints require the Depth-Anything-V2 DINOv2 backbone. Place the
checkpoint at `CKPT/depth_anything_v2_vits.pth`, or point the evaluator to a
custom location:

```bash
export DEPTH_ANYTHING_V2_CKPT=/absolute/path/to/depth_anything_v2_vits.pth
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

Evaluation
----------

Example command for the Goal suite:

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

For a quick smoke test, reduce the number of trials per task:

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint CKPT/M2-VLA-goal \
  --task_suite_name libero_goal \
  --use_pro_version True \
  --num_trials_per_task 1
```

Use the checkpoint table above to switch between LIBERO suites.

Upstream Attribution
--------------------

This release is developed from the VLA-Adapter codebase and keeps the
Prismatic/OpenVLA-style model loading path used by VLA-Adapter. Please also
acknowledge VLA-Adapter when this code is useful for your work, following the
citation guidance in the upstream repository:

- VLA-Adapter: https://github.com/OpenHelix-Team/VLA-Adapter

License
-------

This repository follows the license terms included in `LICENSE`.