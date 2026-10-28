#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/macos/.build"
APP="$BUILD_DIR/MLX Serve.app"
OUTPUT="$BUILD_DIR/MLX-Serve-experimental.dmg"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo 'An Apple Silicon Mac is required.' >&2
    exit 1
fi
if [[ ! -d "$APP/Contents/Resources/runtime" ]]; then
    echo 'Build the complete app first with ./macos/build-app.sh.' >&2
    exit 1
fi
codesign --verify --deep --strict "$APP"
STAGING="$(mktemp -d "$BUILD_DIR/dmg-stage.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
ditto "$APP" "$STAGING/MLX Serve.app"
ln -s /Applications "$STAGING/Applications"
hdiutil create -quiet -fs HFS+ -volname 'MLX Serve' -srcfolder "$STAGING" \
    -format UDZO -ov "$OUTPUT"
hdiutil verify -quiet "$OUTPUT"
echo "Created $OUTPUT"
echo 'Experimental only: the app has not been Developer ID signed or notarized.'
