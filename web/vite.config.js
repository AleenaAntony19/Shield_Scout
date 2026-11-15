import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Dev: the API runs on 127.0.0.1:8000 (python -m shield.api); prod: FastAPI serves web/dist itself.
// SHIELD_API_URL points the dev proxy at an API on another host or port (e.g. when SHIELD_PORT is changed).
export default defineConfig({
  plugins: [vue()],
  server: { port: 5173, proxy: { '/api': process.env.SHIELD_API_URL || 'http://127.0.0.1:8000' } }
})
