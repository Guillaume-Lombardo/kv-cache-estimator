#!/usr/bin/env bash
# Test the production image with its bundled catalogue and a read-only YAML override.
#
# Usage: smoke-container.sh IMAGE [BASE_PATH]
#   BASE_PATH must match the KV_ESTIMATOR_BASE_PATH the image was built with: empty for the
#   root (default) or a prefix such as /kv-estimator.
# Environment:
#   CONTAINER_ENGINE  docker (default) or podman.
#   KV_SMOKE_BROWSER  auto (default: run the browser check when a browser is found),
#                     require (fail without a browser) or skip.
set -euo pipefail
image="${1:?Usage: smoke-container.sh IMAGE [BASE_PATH]}"
base_path="${2:-}"
engine="${CONTAINER_ENGINE:-docker}"
browser_mode="${KV_SMOKE_BROWSER:-auto}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/.tmp"
test_dir="$(mktemp -d "$repo_dir/.tmp/smoke-container.XXXXXX")"
container_id=''
cleanup() {
  if [[ -n "$container_id" ]]; then
    "$engine" logs "$container_id" || true
    "$engine" rm -f "$container_id" >/dev/null || true
  fi
  rm -rf "$test_dir"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

if [[ -n "$base_path" && ! "$base_path" =~ ^(/[A-Za-z0-9._~-]+)+$ ]]; then
  fail "BASE_PATH must be empty or start with / and have no trailing slash (got '$base_path')."
fi
echo "== Testing $image with base path '${base_path:-/}' using $engine"

start_container() {
  container_id="$("$engine" run -d -p 127.0.0.1::3000 "$@" "$image")"
  port="$("$engine" port "$container_id" 3000/tcp | head -1 | awk -F: '{print $NF}')"
  origin="http://127.0.0.1:$port"
}

stop_container() {
  "$engine" rm -f "$container_id" >/dev/null
  container_id=''
}

# Fetch a page, following redirects (/kv-estimator/ redirects to /kv-estimator).
# Usage: fetch_page OUTPUT_FILE PATH [CURL_ARGS...]; prints the final URL.
fetch_page() {
  local output="$1" path="$2"
  shift 2
  curl --fail --silent --show-error --location --max-redirs 5 \
    --output "$output" --write-out '%{url_effective}' "$@" "$origin$path"
}

wait_for_page() {
  for _ in {1..60}; do
    if fetch_page "$test_dir/page.html" "$base_path/" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail 'Container did not become ready within 60 seconds.'
}

# The error page also returns HTTP 200, so check for the estimator itself.
check_estimator() {
  local file="$1" lang="$2" label
  if [[ "$lang" == en ]]; then label='aria-label="Results"'; else label='aria-label="Résultats"'; fi
  grep -q "<html lang=\"$lang\"" "$file" || fail "expected <html lang=\"$lang\">."
  grep -q "$label" "$file" || fail "the $lang estimator ($label) is missing from the page."
}

check_pages() {
  local final_url
  # The prefixed root may redirect; it must end on a URL that keeps the prefix.
  final_url="$(fetch_page "$test_dir/page.html" "$base_path/")"
  case "$final_url" in
    "$origin$base_path" | "$origin$base_path/" | "$origin$base_path"[?/]*) ;;
    *) fail "$base_path/ ended on $final_url, outside the base path." ;;
  esac
  check_estimator "$test_dir/page.html" fr
  fetch_page "$test_dir/page-en.html" "$base_path/" -H 'Accept-Language: en-US,en;q=0.9' > /dev/null
  check_estimator "$test_dir/page-en.html" en
  fetch_page "$test_dir/page-cookie.html" "$base_path" -H 'Accept-Language: fr' -H 'Cookie: kv-lang=en' > /dev/null
  check_estimator "$test_dir/page-cookie.html" en
}

# Extract the CSS and JavaScript URLs from the HTML, then check that each one is served
# under the base path with the right status and Content-Type.
check_assets() {
  local urls url kind content_type headers css=0 js=0
  urls="$(grep -oE '(href|src)="[^"]+\.(css|js)(\?[^"]*)?"' "$test_dir/page.html" \
    | sed -E 's/^(href|src)="//; s/"$//; s/&amp;/\&/g' | sort -u)"
  [[ -n "$urls" ]] || fail 'no CSS or JavaScript URL found in the HTML.'
  while IFS= read -r url; do
    [[ "$url" == "$base_path/_next/static/"* ]] || fail "asset $url is not under $base_path/_next/static/."
    case "$url" in
      *.css | *.css\?*) kind=css; css=$((css + 1)) ;;
      *) kind=js; js=$((js + 1)) ;;
    esac
    headers="$(curl --silent --show-error --output /dev/null \
      --write-out '%{http_code} %{content_type}' "$origin$url")"
    [[ "${headers%% *}" == 200 ]] || fail "$url returned HTTP ${headers%% *}."
    content_type="${headers#* }"
    if [[ "$kind" == css ]]; then
      [[ "$content_type" == text/css* ]] || fail "$url has Content-Type '$content_type'."
    else
      [[ "$content_type" =~ ^(application|text)/javascript ]] || fail "$url has Content-Type '$content_type'."
    fi
    echo "  $headers $url"
  done <<< "$urls"
  ((css > 0)) || fail 'no stylesheet found in the HTML.'
  ((js > 0)) || fail 'no script found in the HTML.'
  echo "Checked $css stylesheet(s) and $js script(s)."

  if [[ -n "$base_path" ]]; then
    # Without the prefix, neither the page nor the assets may be served.
    url="$(head -1 <<< "$urls")"
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "$origin${url#"$base_path"}")" == 404 ]] \
      || fail "${url#"$base_path"} is served without the $base_path prefix."
    headers="$(curl --silent --show-error --output "$test_dir/unprefixed.html" \
      --write-out '%{http_code}' "$origin/")"
    [[ "$headers" == 404 ]] || fail "the unprefixed root returned HTTP $headers, expected 404."
    if grep -q 'aria-label="Résultats"' "$test_dir/unprefixed.html"; then
      fail "the estimator is also served at / although the image uses $base_path."
    fi
  fi
}

check_image() {
  local built_path
  # The standalone server reads the base path compiled into the build, not the environment.
  built_path="$("$engine" exec "$container_id" \
    node -p "require('/app/.next/required-server-files.json').config.basePath")"
  [[ "$built_path" == "$base_path" ]] \
    || fail "the image was built with base path '$built_path', expected '$base_path'. Rebuild it with --build-arg KV_ESTIMATOR_BASE_PATH=$base_path."
  [[ "$("$engine" exec "$container_id" id -u)" != 0 ]] || fail 'the container runs as root.'
}

check_browser() {
  [[ "$browser_mode" == skip ]] && { echo 'Browser check skipped (KV_SMOKE_BROWSER=skip).'; return 0; }
  if ! (cd "$repo_dir" && node -e "require.resolve('playwright-core')") > /dev/null 2>&1; then
    [[ "$browser_mode" == require ]] && fail 'playwright-core is not installed; run pnpm install.'
    echo 'Browser check skipped: playwright-core is not installed.'
    return 0
  fi
  KV_SMOKE_BROWSER="$browser_mode" node "$repo_dir/scripts/smoke-browser.mjs" "$origin" "$base_path"
}

start_container
check_image
wait_for_page
check_pages
check_assets
check_browser
stop_container

# Change a scalar catalogue name without corrupting localized name mappings.
sed 's/^    name: H100 SXM$/    name: CI_CUSTOM_CATALOGUE_H100/' \
  "$repo_dir/config/estimator.yaml" > "$test_dir/catalogue.yaml"
chmod 755 "$test_dir"
chmod 644 "$test_dir/catalogue.yaml"
start_container \
  --mount "type=bind,src=$test_dir/catalogue.yaml,dst=/config/catalogue.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/catalogue.yaml
wait_for_page
check_estimator "$test_dir/page.html" fr
grep -q 'CI_CUSTOM_CATALOGUE_' "$test_dir/page.html" || fail 'the mounted catalogue was not used.'
[[ "$("$engine" exec "$container_id" id -u)" != 0 ]] || fail 'the container runs as root.'
echo "Container smoke tests passed for base path '${base_path:-/}'."
