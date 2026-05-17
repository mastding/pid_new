import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

function vendorChunkName(id: string) {
  const normalized = id.replace(/\\/g, '/');
  const marker = '/node_modules/';
  const markerIndex = normalized.lastIndexOf(marker);
  if (markerIndex < 0) return undefined;
  const packagePath = normalized.slice(markerIndex + marker.length);
  const parts = packagePath.split('/');
  const packageName = parts[0].startsWith('@') ? `${parts[0]}/${parts[1]}` : parts[0];
  if (
    packageName === 'react'
    || packageName === 'react-dom'
    || packageName === 'scheduler'
    || packageName === 'react-is'
    || packageName === 'react-router'
    || packageName === 'react-router-dom'
    || packageName === 'use-sync-external-store'
    || packageName === '@remix-run/router'
  ) {
    return 'vendor-react';
  }
  if (packageName === 'antd') return 'vendor-antd';
  if (packageName === '@ant-design/icons' || packageName === '@ant-design/icons-svg') return 'vendor-icons';
  if (packageName === '@ant-design/charts' || packageName === '@ant-design/plots' || packageName === '@ant-design/graphs') return 'vendor-charts';
  if (packageName.startsWith('@ant-design/pro-') || packageName === '@ant-design/pro-components') return 'vendor-pro-components';
  if (packageName.startsWith('@ant-design/')) return 'vendor-antd-deps';
  if (packageName.startsWith('@rc-component/') || packageName.startsWith('rc-')) return 'vendor-antd-deps';
  if (packageName.startsWith('@antv/')) return 'vendor-antv';
  if (packageName.startsWith('d3-')) return 'vendor-d3';
  if (packageName === 'axios' || packageName === 'dayjs') return 'vendor-utils';
  if (packageName === 'lodash' || packageName === 'lodash-es' || packageName.startsWith('lodash.')) return 'vendor-lodash';
  if (packageName === 'html2canvas') return 'vendor-html2canvas';
  return 'vendor-misc';
}

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 1100,
    rollupOptions: {
      output: {
        manualChunks: vendorChunkName,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:4455',
        changeOrigin: true,
      },
    },
  },
});
