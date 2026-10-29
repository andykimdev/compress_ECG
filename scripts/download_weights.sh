#!/usr/bin/env bash
# Download the public EchoNext-Mini code and the pretrained teacher weights.
#
# Purpose:
# - Give every machine the same files, pinned to fixed versions.
# - Check each file against its SHA-256 hash and stop on any mismatch.
#
# Special Notes:
# - Run from the repository root: bash scripts/download_weights.sh
# - Files that already match their hash are skipped, so reruns are cheap.
# - The EchoNext ECG data is not downloaded here. It needs a signed PhysioNet agreement.

set -euo pipefail

INTROECG_COMMIT=15233e9392e9dc10c136a5624abf10199207bce9
ECG_FM_REV=584219ea492cdeef2e19ffbdf9c6ecc874ba427e
ECGFOUNDER_REV=d9b1793951b2342f5f7e84f1ac03cd37f8a08724

HF=https://huggingface.co
ECG_FM_DIR=checkpoints/pretrained/ecg_fm
ECGFOUNDER_DIR=checkpoints/pretrained/ecgfounder
MINI_WEIGHTS="external/IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt"

sha256() {
    # macOS ships shasum and Linux ships sha256sum.
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

verify() {
    local path=$1 expected=$2
    local actual
    actual=$(sha256 "$path")
    if [[ "$actual" != "$expected" ]]; then
        echo "Hash mismatch for $path: expected $expected, got $actual" >&2
        exit 1
    fi
}

fetch() {
    local url=$1 path=$2 expected=$3
    if [[ -f "$path" && "$(sha256 "$path")" == "$expected" ]]; then
        echo "ok  $path (already present)"
        return
    fi
    mkdir -p "$(dirname "$path")"
    echo "get $path"
    # Check under a temporary name so an interrupted or corrupt download never takes the real name.
    curl -fL --retry 3 -o "$path.part" "$url"
    verify "$path.part" "$expected"
    mv "$path.part" "$path"
    echo "ok  $path"
}

if [[ ! -d external/IntroECG/.git ]]; then
    git clone https://github.com/PierreElias/IntroECG.git external/IntroECG
fi
git -C external/IntroECG fetch --quiet origin
git -C external/IntroECG checkout --quiet "$INTROECG_COMMIT"
verify "$MINI_WEIGHTS" aac57aad0ec9763021c6264e89c74e3d78f8b711159b1880e5bd4515d5e21cd6
echo "ok  $MINI_WEIGHTS"

fetch "$HF/wanglab/ecg-fm/resolve/$ECG_FM_REV/mimic_iv_ecg_physionet_pretrained.yaml" \
    "$ECG_FM_DIR/mimic_iv_ecg_physionet_pretrained.yaml" \
    710f5fdc54c31fd88c82d3ec1ad9c39c82a7848201fa6afd84074d5b71b726ab
fetch "$HF/wanglab/ecg-fm/resolve/$ECG_FM_REV/mimic_iv_ecg_physionet_pretrained.pt" \
    "$ECG_FM_DIR/mimic_iv_ecg_physionet_pretrained.pt" \
    4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e
fetch "$HF/PKUDigitalHealth/ECGFounder/resolve/$ECGFOUNDER_REV/config.json" \
    "$ECGFOUNDER_DIR/config.json" \
    11f2874fbc95b25fec2152980b8c7bdf79e3872f92bf8218a810c5c69008d776
fetch "$HF/PKUDigitalHealth/ECGFounder/resolve/$ECGFOUNDER_REV/12_lead_ECGFounder.pth" \
    "$ECGFOUNDER_DIR/12_lead_ECGFounder.pth" \
    ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997

echo "All weights are in place."
