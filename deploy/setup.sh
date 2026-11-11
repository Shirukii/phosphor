#!/usr/bin/env bash
# One-time setup on the server, as a user with sudo (asks for the password):
#   - a `phosphor` system user for the service to run as (it owns nothing, it only reads the app)
#   - /opt/phosphor, owned by you, so deploy.sh can update it without sudo
#   - the systemd unit, and the Caddy site (appended to /etc/caddy/Caddyfile, after a backup)
# Then run deploy.sh. Needs DNS for the site's name pointing here first, so Caddy gets a certificate.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
me=$(id -un)
site=$(awk 'NR>1 && /\{$/ {print $1; exit}' "$here/phosphor.caddy")

id phosphor >/dev/null 2>&1 || sudo useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin phosphor
sudo install -d -o "$me" -g "$me" -m 755 /opt/phosphor
sudo install -m 644 "$here/phosphor.service" /etc/systemd/system/phosphor.service
sudo systemctl daemon-reload
sudo systemctl enable phosphor.service

if ! grep -q "^$site {" /etc/caddy/Caddyfile; then
  sudo cp -a /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak-phosphor-$(date +%F-%H%M%S)"
  { echo; cat "$here/phosphor.caddy"; } | sudo tee -a /etc/caddy/Caddyfile >/dev/null
  sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
  sudo systemctl reload caddy
  echo "added $site to Caddy"
else
  echo "$site is in the Caddyfile already: left as it is"
fi
echo "set up. Now: $here/deploy.sh"
