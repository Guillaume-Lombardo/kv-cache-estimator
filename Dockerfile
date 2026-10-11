# Production image using Next.js standalone output.
# Configuration is read at startup. Mount a custom file and set KV_ESTIMATOR_CONFIG to its path.
#   docker build -t kv-cache-estimator .
#   docker run -p 3000:3000 -v $PWD/my-catalogue.yaml:/config/estimator.yaml \
#     -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml kv-cache-estimator
FROM node:22-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm run check-config && pnpm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/config ./config
EXPOSE 3000
USER node
CMD ["node", "server.js"]
