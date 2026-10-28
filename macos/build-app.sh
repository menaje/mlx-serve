#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAC_DIR="$ROOT/macos"
BUILD_DIR="$MAC_DIR/.build"
RUNTIME="$BUILD_DIR/portable-python"
ARCHIVE="$BUILD_DIR/python-3.14.10.tar.gz"
APP="$BUILD_DIR/MLX Serve.app"
URL='https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.14.10%2B20261026-aarch64-apple-darwin-install_only_stripped.tar.gz'
SHA='95eb8c87d16af37c54e9765e31d8c56c0d8ffa9968ea0f704f73a9b83f84e9f9'

if [[ "$(uname -m)" != "arm64" ]]; then
    echo 'An Apple Silicon Mac is required.' >&2
    exit 1
fi
mkdir -p "$BUILD_DIR"
if [[ "${1:-}" != '--development' ]]; then
    if [[ ! -f "$RUNTIME/.mlx-serve-runtime-complete" ]]; then
        curl --fail --location --retry 2 "$URL" --output "$ARCHIVE"
        printf '%s  %s\n' "$SHA" "$ARCHIVE" | shasum -a 256 --check
        rm -rf "$RUNTIME" "$BUILD_DIR/python"
        tar -xzf "$ARCHIVE" -C "$BUILD_DIR"
        mv "$BUILD_DIR/python" "$RUNTIME"
    fi
    # Reconcile the pins on every build; a cached runtime may predate changes
    # to requirements-runtime.txt.
    "$RUNTIME/bin/python3" -m pip install --only-binary=:all: --no-deps \
        -r "$MAC_DIR/requirements-runtime.txt"
    touch "$RUNTIME/.mlx-serve-runtime-complete"
    # Reinstall the current checkout every time, including after source changes.
    "$RUNTIME/bin/python3" -m pip install --no-deps --force-reinstall "$ROOT"
    "$RUNTIME/bin/python3" -m pip check
fi

swift build --package-path "$MAC_DIR" -c release
BIN_DIR="$(swift build --package-path "$MAC_DIR" -c release --show-bin-path)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$MAC_DIR/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN_DIR/MLXServeApp" "$APP/Contents/MacOS/MLXServeApp"
if [[ "${1:-}" != '--development' ]]; then
    ditto "$RUNTIME" "$APP/Contents/Resources/runtime"
fi
codesign --force --deep --sign - "$APP"
echo "$APP"
