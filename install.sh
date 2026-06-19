#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="$HERMES_HOME/plugins/model-providers/cursor-agent"
SOURCE="$ROOT/model-providers/cursor-agent"

if [[ ! -f "$SOURCE/plugin.yaml" ]]; then
  echo "error: plugin source not found at $SOURCE" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET")"
if [[ -e "$TARGET" ]]; then
  backup="$TARGET.bak.$(date +%Y%m%d_%H%M%S)"
  echo "Backing up existing plugin to $backup"
  mv "$TARGET" "$backup"
fi

cp -a "$SOURCE" "$TARGET"
echo "Installed cursor-agent provider to $TARGET"
echo
echo "Next steps:"
echo "  1. Install Cursor Agent CLI: curl https://cursor.com/install -fsS | bash"
echo "  2. Set CURSOR_API_KEY in $HERMES_HOME/.env"
echo "  3. Run: hermes model   (pick cursor-agent)"
echo "  4. Verify: hermes doctor"
