#!/usr/bin/env bash
# (Re)deploys the web interface from this checkout: pulls, builds the page, copies the app to
# /opt/phosphor/app, updates its venv, and restarts the service (sudo asks for the password).
# Run on the server, in the checkout (setup.sh first, once). Nothing secret is involved.
set -euo pipefail
cd "$(dirname "$0")/.."
# haruka keeps its node per user; anywhere else, node on the PATH will do
[ -d ~/.local/share/laika-node/current/bin ] && export PATH=~/.local/share/laika-node/current/bin:$PATH

git pull -q --ff-only
npm ci --no-audit --no-fund --loglevel=error
npm run -s build:web

app=/opt/phosphor/app
mkdir -p "$app.new"
cp -a src tools web deploy package.json "$app.new/"
rm -rf "$app.new/web/app.ts" "$app.new/web/dist/"*.map
find "$app.new" -name __pycache__ -prune -exec rm -rf {} +
chmod -R a+rX "$app.new"
[ -d /opt/phosphor/venv ] || python3 -m venv /opt/phosphor/venv
/opt/phosphor/venv/bin/pip install -q --disable-pip-version-check -r deploy/requirements.txt
chmod -R a+rX /opt/phosphor/venv
rm -rf "$app.old"; [ -d "$app" ] && mv "$app" "$app.old"; mv "$app.new" "$app"; rm -rf "$app.old"

sudo systemctl restart phosphor.service
sleep 2
if curl -fsS -o /dev/null http://127.0.0.1:8340/api/config; then
  echo "deployed $(git log --oneline -1)"
else
  echo "phosphor did not come up: journalctl -u phosphor -n 50" >&2; exit 1
fi
