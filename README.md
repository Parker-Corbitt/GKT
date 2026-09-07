# GKT

The implementation of the paper [Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network](https://dl.acm.org/doi/10.1145/3350546.3352513).

The architecture of the GKT is as follows:

![GKT architecture](gkt_architecture.png)

## Setup

Python 3.12 is supported on CPU and CUDA-capable Linux systems where the installed PyTorch build supports CUDA. macOS uses CPU execution.

Create the project environment and install the tested dependency set with:

```bash
./setup.sh
source .gkt/bin/activate
```

The dependency pins in `requirements.txt` provide CPython 3.12 wheels for NumPy, pandas, SciPy, scikit-learn, and PyTorch. Dataset grouping uses explicit user iteration and does not depend on pandas' historical `GroupBy.apply` behavior.

## Training the model

Use the `train.py` script to train the model. To train the GKT model on ASSISTments2009-2010 skill-builder dataset, simply use:

```bash
python train.py --data-file=skill_builder_data.csv --model=GKT --graph-type=Dense
```

We also provide the baseline, i.e. Deep Knowledge Tracing(DKT) for performance comparison. To train the DKT model on ASSISTments2009-2010 skill-builder dataset, simply use:

```bash
python train.py --data-file=skill_builder_data.csv --model=DKT
```

The relevant options are `--data-dir` and `--save-dir`. Use `--no-cuda` for an explicit CPU run, or `--bias false`, `--binary false`, and `--shuffle false` for boolean settings that need to be disabled.

## Athena memory inside GKT

The project includes an Athena-inspired multiple-trace memory based on
Briglia et al., [*Modeling an enactivist multiple-trace memory. ATHENA: A
fractal model of human memory*](https://doi.org/10.1016/j.jmp.2017.12.002).
When enabled, GKT writes each answered concept as a trace, retrieves traces by
cue similarity and power-law recency, and fuses the retrieved context into the
GKT hidden state before graph propagation and prediction.

Enable it with:

```bash
python train.py --model=GKT --graph-type=Dense --memory=athena --no-cuda
```

Tune `--max-traces` and `--memory-decay` to control memory capacity and
forgetting. The default `--memory=none` preserves the original GKT behavior.

Run the compatibility smoke tests with:

```bash
python -m unittest discover -s tests
```
