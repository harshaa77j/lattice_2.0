import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development (npm run dev), API calls are forwarded to your local
// FastAPI server, so the frontend and backend behave as if on one address.
const api = "http://localhost:8000";
const apiPaths = ["/jobs", "/admin", "/status", "/health", "/register", "/heartbeat", "/nodes"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiPaths.map((p) => [p, api])),
  },
});
