# Image de production (sortie "standalone" de Next.js).
# Le fichier de configuration est lu au démarrage : montez le vôtre et pointez KV_ESTIMATOR_CONFIG dessus.
#   docker build -t kv-cache-estimator .
#   docker run -p 3000:3000 -v $PWD/mon-catalogue.yaml:/config/estimator.yaml \
#     -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml kv-cache-estimator
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run check-config && npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/config ./config
EXPOSE 3000
USER node
CMD ["node", "server.js"]
