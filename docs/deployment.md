# Deployment

How to run the estimator in a container, behind a reverse proxy under a URL prefix, and how releases publish images. Back to the [README](../README.md).

## Language and theme

The interface is available in French and English, with a light, dark or system theme. The buttons at the top right switch both; the choice is stored in cookies (`kv-lang`, `kv-theme`) so the server renders the right language and theme directly. Without a cookie, the language follows the browser's `Accept-Language` header (French by default).

## Ready-to-run container

Once a release has completed its publication workflow, its image is available at `ghcr.io/guillaume-lombardo/kv-cache-estimator`. Images include the default catalogue and support `linux/amd64` and `linux/arm64`. No GPU is required to run the estimator.

```bash
# latest follows the most recently published stable release.
docker run --rm -p 3000:3000 \
  ghcr.io/guillaume-lombardo/kv-cache-estimator:latest
```

Open <http://localhost:3000>. For reproducible deployments, replace `latest` with a published release tag such as `v0.1.0`, or use the image digest recorded in the workflow summary. `v0.1.0` is an example, not a claim that this release already exists.

Mount a custom catalogue read-only:

```bash
docker run --rm -p 3000:3000 \
  --mount "type=bind,src=$PWD/my-catalogue.yaml,dst=/config/estimator.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml \
  ghcr.io/guillaume-lombardo/kv-cache-estimator:latest
```

The image runs as the non-root `node` user. Mounted files must be readable by that user. The default port is 3000; `PORT`, `HOSTNAME` and `KV_ESTIMATOR_CONFIG` can be overridden at runtime. `KV_ESTIMATOR_BASE_PATH` is a build argument; see [Serving under a URL prefix](#serving-under-a-url-prefix).

To build and test an image locally:

```bash
docker build -t kv-cache-estimator:local .
bash scripts/smoke-container.sh kv-cache-estimator:local
```

## Serving under a URL prefix

By default the application is served at the root (`/`). To expose it under a prefix such as `/kv-estimator`, for example behind a reverse proxy that shares a host name with other services, build it with `KV_ESTIMATOR_BASE_PATH`. The value is passed to Next.js as [`basePath`](https://nextjs.org/docs/app/api-reference/config/next-config-js/basePath), so pages, `/_next/static` CSS and JavaScript, and redirects all carry the prefix.

- The value must be empty (root, the default) or a path that starts with `/` and has no trailing slash, such as `/kv-estimator` or `/tools/kv-estimator`. Any other value, such as `/kv-estimator/`, `kv-estimator` or `/`, stops the build with an `Invalid KV_ESTIMATOR_BASE_PATH` error.
- The prefix is compiled into the standalone server at build time. Setting `KV_ESTIMATOR_BASE_PATH` when starting a container has no effect: **changing the prefix requires rebuilding the image.** The published GHCR images are built for the root.
- With a prefix, `/kv-estimator` serves the estimator, `/kv-estimator/` redirects to it (HTTP 308, relative `Location: /kv-estimator`), and `/` returns 404.

### Build and run without a container

```bash
KV_ESTIMATOR_BASE_PATH=/kv-estimator pnpm run build
KV_ESTIMATOR_BASE_PATH=/kv-estimator pnpm start    # http://localhost:3000/kv-estimator
```

### Build and run with Docker

```bash
# Root (default): http://localhost:3000/
docker build -t kv-cache-estimator:root .
docker run --rm -p 3000:3000 kv-cache-estimator:root

# Under /kv-estimator: http://localhost:3000/kv-estimator
docker build --build-arg KV_ESTIMATOR_BASE_PATH=/kv-estimator -t kv-cache-estimator:kv-estimator .
docker run --rm -p 3000:3000 kv-cache-estimator:kv-estimator
```

### Build and run with Podman

```bash
# Root (default): http://localhost:3000/
podman build -t kv-cache-estimator:root .
podman run --rm -p 3000:3000 localhost/kv-cache-estimator:root

# Under /kv-estimator: http://localhost:3000/kv-estimator
podman build --build-arg KV_ESTIMATOR_BASE_PATH=/kv-estimator -t kv-cache-estimator:kv-estimator .
podman run --rm -p 3000:3000 localhost/kv-cache-estimator:kv-estimator
```

A custom catalogue is mounted the same way in both modes (add `:Z` or `,relabel=private` to the mount on SELinux hosts with Podman):

```bash
docker run --rm -p 3000:3000 \
  --mount "type=bind,src=$PWD/my-catalogue.yaml,dst=/config/estimator.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml \
  kv-cache-estimator:kv-estimator
```

The images still run as the non-root `node` user. To check which prefix an image was built with:

```bash
docker run --rm kv-cache-estimator:kv-estimator \
  node -p "require('/app/.next/required-server-files.json').config.basePath"
# /kv-estimator   (empty output means the root)
```

### Health checks

Under a prefix, the root no longer answers with HTTP 200, so point health checks at the prefixed path. Probe `/kv-estimator` without a trailing slash to get a direct 200 rather than a 308 redirect (or make sure the probe follows redirects). The image is based on Alpine and has no `curl`; use BusyBox `wget` or Node.js:

```bash
# Docker or Podman
docker run -d -p 3000:3000 \
  --health-cmd 'wget -q --spider http://127.0.0.1:3000/kv-estimator || exit 1' \
  --health-interval 30s \
  kv-cache-estimator:kv-estimator
```

```yaml
# Docker Compose
healthcheck:
  test: ["CMD", "node", "-e", "fetch('http://127.0.0.1:3000/kv-estimator').then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))"]
  interval: 30s
  timeout: 5s
  retries: 3

# Kubernetes
readinessProbe:
  httpGet:
    path: /kv-estimator
    port: 3000
```

At the root, use `http://127.0.0.1:3000/` and `path: /` instead. `podman run` accepts the same `--health-cmd` and `--health-interval` options.

### Traefik

Route the prefix to the Next.js server **without stripping it**: the server expects to receive `/kv-estimator/...` paths. Do not attach a `StripPrefix` middleware to this router; with one, Next.js receives paths without the prefix and returns 404.

Docker or Podman labels (Traefik Docker provider):

```yaml
services:
  kv-estimator:
    image: kv-cache-estimator:kv-estimator
    labels:
      - traefik.enable=true
      - traefik.http.routers.kv-estimator.rule=PathPrefix(`/kv-estimator`)
      - traefik.http.routers.kv-estimator.entrypoints=websecure
      - traefik.http.services.kv-estimator.loadbalancer.server.port=3000
      # No traefik.http.routers.kv-estimator.middlewares=...stripprefix...
```

File provider:

```yaml
http:
  routers:
    kv-estimator:
      rule: PathPrefix(`/kv-estimator`)
      entryPoints: [websecure]
      service: kv-estimator
      # No StripPrefix middleware.
  services:
    kv-estimator:
      loadBalancer:
        servers:
          - url: http://kv-estimator:3000
```

``PathPrefix(`/kv-estimator`)`` matches both `/kv-estimator` and `/kv-estimator/...`. To exclude unrelated paths such as `/kv-estimator-old`, use ``Path(`/kv-estimator`) || PathPrefix(`/kv-estimator/`)`` instead. Add a host rule (``Host(`example.com`) && ...``) when Traefik serves several host names.

After deploying, check that the page and its redirect stay under the prefix and that the assets load:

```bash
curl -sI https://example.com/kv-estimator/ | grep -i -E '^(HTTP|location)'   # 308, location: /kv-estimator
curl -sL https://example.com/kv-estimator/ | grep -o '/kv-estimator/_next/static/[^"]*\.css' | head -1
```

The second command must print a stylesheet path under `/kv-estimator/_next/static/`; requesting it must return HTTP 200 with `Content-Type: text/css`.

### Smoke tests

`scripts/smoke-container.sh IMAGE [BASE_PATH]` tests an image built for the given prefix (empty for the root):

```bash
bash scripts/smoke-container.sh kv-cache-estimator:root
bash scripts/smoke-container.sh kv-cache-estimator:kv-estimator /kv-estimator
CONTAINER_ENGINE=podman bash scripts/smoke-container.sh localhost/kv-cache-estimator:kv-estimator /kv-estimator
```

For each mode it checks that the image was built with the expected prefix and runs as a non-root user, that the estimator is served in French and English (`Accept-Language` and the `kv-lang` cookie), that `BASE_PATH/` stays under the prefix after any redirect, and that every CSS and JavaScript URL in the HTML is under `BASE_PATH/_next/static/` and returns HTTP 200 with a CSS or JavaScript `Content-Type`. With a prefix, it also checks that neither the page nor the assets are served without it. It then restarts the image with a mounted catalogue and checks that the catalogue is used.

When `playwright-core` is installed (`pnpm install`) and a Chromium or Google Chrome browser is available, `scripts/smoke-browser.mjs` also opens the page, checks that the stylesheet is applied, toggles the language to confirm that the JavaScript has hydrated, and reloads to confirm that the server renders the chosen language. Set `KV_SMOKE_BROWSER=require` to fail without a browser, `KV_SMOKE_BROWSER=skip` to skip the check, and `KV_SMOKE_BROWSER_PATH` to use a specific browser executable.

## Releases and image publication

The GitHub Actions workflow validates the catalogue, runs TypeScript checks and tests, builds the application, then builds and smoke-tests the production container twice: at the root and under `/kv-estimator`. The smoke tests check both the bundled catalogue and a mounted YAML override, the French and English pages, the CSS and JavaScript assets and their loading in a browser, and verify that the container runs without root privileges. Published images are built for the root.

Publishing a GitHub Release triggers image publication after those checks pass:

1. Merge the intended changes into `main`.
2. Create a release with a tag such as `v0.1.0` or `v0.2.0-rc.1`, pointing to the intended commit. Use the GitHub prerelease flag for release candidates.
3. Publish the release and wait for the `publish` job to succeed.
4. Use the version tag or digest from the workflow summary.

Stable releases publish their exact tag and `latest`. Prereleases publish only their exact tag and never replace `latest`. Draft releases and ordinary branch pushes do not publish images. Release tags are the image version authority; `package.json` is not used to generate image tags. Avoid republishing a version tag; use digests when immutability matters.

The workflow authenticates to GHCR with `GITHUB_TOKEN` and grants `packages: write` only to the publication job. It attaches OCI metadata, build provenance and an SBOM. The runtime smoke test runs on amd64; the publication job builds both architectures.

After the first publication, check the package's visibility in GitHub Packages. GHCR may create a private package initially. Set it to public for anonymous pulls, or authenticate Docker with an account that can read it. Repository or organization policies must allow Actions to publish packages.
