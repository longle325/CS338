import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_PROXY_TARGET || "http://127.0.0.1:8010",
        changeOrigin: true,
      },
      "/artifacts": {
        target: process.env.VITE_BACKEND_PROXY_TARGET || "http://127.0.0.1:8010",
        changeOrigin: true,
      },
      "/demo-examples": {
        target: process.env.VITE_BACKEND_PROXY_TARGET || "http://127.0.0.1:8010",
        changeOrigin: true,
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
