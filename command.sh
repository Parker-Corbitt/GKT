#!/usr/bin/env bash

set -euo pipefail

python train.py --data-file=skill_builder_data.csv --graph-type=VAE
python train.py --data-file=skill_builder_data.csv --graph-type=MHA
python train.py --data-file=assistment_test15.csv --graph-type=Dense
python train.py --data-file=assistment_test15.csv --graph-type=Transition
python train.py --data-file=assistment_test15.csv --graph-type=PAM