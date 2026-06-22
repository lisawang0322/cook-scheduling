FROM oven/bun:1

WORKDIR /app

# Install deps with lockfile for reproducibility
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Copy source
COPY . .

EXPOSE 8080

# Pass --host so Vite binds to 0.0.0.0 (required inside containers)
CMD ["bun", "run", "dev", "--host"]
