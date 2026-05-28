# OmniTry++ Frontend

This Vite app is wired to the live FastAPI backend in `demo/backend`.

## Run With Live Backend

```bash
conda activate cs338
bash demo/backend/run.sh
```

In another terminal:

```bash
cd demo/frontend
npm install
npm run dev
```

By default, Vite proxies `/api`, `/artifacts`, and `/demo-examples` to `http://127.0.0.1:8010`.
Override the proxy target with:

```bash
VITE_BACKEND_PROXY_TARGET=http://127.0.0.1:8010 npm run dev
```

For a direct backend URL instead of the dev proxy, create `demo/frontend/.env.local`:

```bash
VITE_TRYON_API_BASE_URL=http://127.0.0.1:8010
```

## Mock Mode

Use mock responses when the GPU backend is not running:

```bash
VITE_TRYON_USE_MOCK=true npm run dev
```
