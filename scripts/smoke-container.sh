#!/usr/bin/env bash
# Test the production image with its bundled catalogue and a read-only YAML override.
set -euo pipefail
image="${1:?Usage: smoke-container.sh IMAGE}"
test_dir="$(mktemp -d)"
container_id=''
cleanup() {
  if [[ -n "$container_id" ]]; then
    docker logs "$container_id" || true
    docker rm -f "$container_id" >/dev/null || true
  fi
  rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_page() {
  local port
  port="$(docker port "$container_id" 3000/tcp | head -1 | awk -F: '{print $NF}')"
  for attempt in {1..60}; do
    if curl --fail --silent "http://127.0.0.1:$port" > "$test_dir/page.html"; then
      return 0
    fi
    sleep 1
  done
  echo 'Container did not become ready within 60 seconds.' >&2
  return 1
}

container_id="$(docker run -d -p 127.0.0.1::3000 "$image")"
wait_for_page
# The error page also returns HTTP 200, so check for the estimator itself.
grep -q 'aria-label="Résultats"' "$test_dir/page.html"
[[ "$(docker exec "$container_id" id -u)" != 0 ]]
docker rm -f "$container_id" >/dev/null
container_id=''

# Change catalogue names so the response proves that the mounted file was loaded.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sed 's/^    name: /    name: CI_CUSTOM_CATALOGUE_/' \
  "$repo_dir/config/estimator.yaml" > "$test_dir/catalogue.yaml"
chmod 755 "$test_dir"
chmod 644 "$test_dir/catalogue.yaml"
container_id="$(docker run -d -p 127.0.0.1::3000 \
  --mount "type=bind,src=$test_dir/catalogue.yaml,dst=/config/catalogue.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/catalogue.yaml "$image")"
wait_for_page
grep -q 'aria-label="Résultats"' "$test_dir/page.html"
grep -q 'CI_CUSTOM_CATALOGUE_' "$test_dir/page.html"
echo 'Container smoke tests passed.'
