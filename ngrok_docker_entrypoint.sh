#!/bin/sh
set -eu

set -- http waste-scanner:8000 --log stdout

if [ -n "${NGROK_DOMAIN:-}" ]; then
  case "$NGROK_DOMAIN" in
    http://*|https://*) public_url="$NGROK_DOMAIN" ;;
    *) public_url="https://$NGROK_DOMAIN" ;;
  esac
  set -- "$@" --url "$public_url"
fi

if [ -n "${NGROK_BASIC_AUTH:-}" ]; then
  escaped_credential=$(printf '%s' "$NGROK_BASIC_AUTH" | sed 's/\\/\\\\/g; s/"/\\"/g')
  cat > /tmp/ngrok-traffic-policy.json <<EOF
{"on_http_request":[{"actions":[{"type":"basic-auth","config":{"credentials":["$escaped_credential"]}}]}]}
EOF
  set -- "$@" --traffic-policy-file /tmp/ngrok-traffic-policy.json
fi

exec ngrok "$@"
