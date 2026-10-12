# Production image using Next.js standalone output.
# Configuration is read at startup. Mount a custom file and set KV_ESTIMATOR_CONFIG to its path.
#   docker build -t kv-cache-estimator .
#   docker run -p 3000:3000 -v $PWD/my-catalogue.yaml:/config/estimator.yaml \
#     -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml kv-cache-estimator
# To serve the application under a URL prefix (e.g. behind a reverse proxy), set it at build time.
# Next.js compiles the prefix into the standalone server, so changing it requires a rebuild:
#   docker build --build-arg KV_ESTIMATOR_BASE_PATH=/kv-estimator -t kv-cache-estimator:prefixed .
FROM node:22-alpine AS build
ARG KV_ESTIMATOR_BASE_PATH=""
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
ENV KV_ESTIMATOR_BASE_PATH=$KV_ESTIMATOR_BASE_PATH
RUN pnpm run check-config && pnpm run build

FROM node:22-alpine
ARG KV_ESTIMATOR_BASE_PATH=""
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
# Informational only: the prefix is already part of the build and cannot be changed at runtime.
LABEL io.github.guillaume-lombardo.kv-cache-estimator.base-path="$KV_ESTIMATOR_BASE_PATH"
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/config ./config
EXPOSE 3000
USER node
CMD ["node", "server.js"]
