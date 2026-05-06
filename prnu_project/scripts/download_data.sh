#!/bin/bash
set -e
mkdir -p data/raw/dresden data/raw/ieee_sp

echo "Downloading Dresden Image Database..."
kaggle datasets download -d micscodes/dresden-image-database -p data/raw/dresden --unzip

echo "Downloading IEEE SP Camera Identification..."
kaggle competitions download -c sp-society-camera-model-identification -p data/raw/ieee_sp
unzip -q -o data/raw/ieee_sp/sp-society-camera-model-identification.zip -d data/raw/ieee_sp

echo "Done. Verify:"
ls data/raw/dresden/
ls data/raw/ieee_sp/
