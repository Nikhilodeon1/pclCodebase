#!/bin/bash
# Source this before running experiments on RunPod:
#   source runpod_env.sh

export MIMIC_DIR=/workspace/physionet.org/files/mimiciv/3.1
export EICU_DIR=/workspace/physionet.org/files/eicu-crd/2.0

# PhysioNet 2019 — set this once you've downloaded it.
# If downloaded via wget --user/--password from physionet.org:
#   wget -r -N -c -np --user=<user> --password=<pass> \
#        https://physionet.org/files/challenge-2019/1.0.0/
# Then the path will be:
export PHYSIONET_DIR=/workspace/physionet2019

# Derive the repo dir from this script's own location so results always land
# in the actual checkout (avoids case-sensitivity mismatches like
# pclCodebase vs pclcodebase on Linux).
_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RESULTS_DIR="$_REPO_DIR/results"
export CHECKPOINT_DIR="$_REPO_DIR/results/checkpoints"

export NUM_WORKERS=8
export PCL_TEST_MODE=0

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

echo "PCL environment set:"
echo "  PHYSIONET_DIR = $PHYSIONET_DIR"
echo "  MIMIC_DIR     = $MIMIC_DIR"
echo "  EICU_DIR      = $EICU_DIR"
echo "  RESULTS_DIR   = $RESULTS_DIR"
echo "  TEST_MODE     = $PCL_TEST_MODE"
