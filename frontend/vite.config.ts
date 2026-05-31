import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  base: "/NLA-Agent/",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        app: resolve(__dirname, "index.html"),
        docs: resolve(__dirname, "docs/index.html")
      }
    }
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    watch: {
      ignored: ["**/uploads/**", "**/__pycache__/**", "**/.pytest_cache/**", "**/dist/**"]
    },
    proxy: {
      "/adk": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/adk/, "")
      }
    }
  }
});
