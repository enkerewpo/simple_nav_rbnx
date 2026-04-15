#!/usr/bin/env bash
# build.sh — simple_nav_rbnx: only codegen, no colcon / docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(dirname "$SCRIPT_DIR")}"

echo "=== simple_nav_rbnx build ==="

if command -v rbnx >/dev/null 2>&1; then
    FLAGS=()
    [[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
    rbnx codegen -p "$PKG_ROOT" "${FLAGS[@]}"
else
    echo "[build] rbnx not in PATH — run rbnx setup from robonix source root first" >&2
    exit 1
fi

mkdir -p "$PKG_ROOT/rbnx-build"
date -Iseconds > "$PKG_ROOT/rbnx-build/.rbnx-built"
echo "=== simple_nav_rbnx build complete ==="
