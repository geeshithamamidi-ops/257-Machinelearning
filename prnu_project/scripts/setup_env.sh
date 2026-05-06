#!/bin/bash
set -e
cd "$(dirname "$0")/.."
conda env create -f environment.yml || conda env update -f environment.yml
echo "Activate with: conda activate prnu_project"
