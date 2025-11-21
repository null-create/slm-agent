# SLM Agent UI

A modern, TypeScript-based web interface for interacting with the SLM Agent model.

## Features

- 🎨 Modern, responsive UI with gradient design
- 💬 Real-time chat interface with message history
- 📝 Rich markdown rendering (code blocks, formatting, lists)
- ⚙️ Configurable generation parameters
- 🔧 Tool usage tracking and metadata display
- 🔌 Live server connection monitoring
- 📱 Session management

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Lucide React** for icons

## Prerequisites

- Node.js 18+ and npm
- SLM Agent backend server running (default: `http://localhost:8000`)

## Installation

```bash
# From the ui directory
npm install
```

## Development

Start the development server with hot reload:

```bash
npm run dev
```

The UI will be available at `http://localhost:3000`. The Vite dev server is configured to proxy API requests to `http://localhost:8000`, so make sure your backend server is running.

## Building for Production

Build the optimized production bundle:

```bash
npm run build
```

The built files will be in the `dist/` directory. You can preview the production build:

```bash
npm run preview
```

## Configuration

### Backend Server URL

The default backend server URL is `http://localhost:8000`. You can change this:

1. **During development**: Edit `vite.config.ts` to update the proxy target
2. **At runtime**: Use the Settings panel in the UI to update the server URL

### Generation Parameters

Adjust model generation parameters through the Settings panel (gear icon):

- **Temperature** (0.0 - 2.0): Controls randomness. Lower = more focused, higher = more creative
- **Max Tokens** (128 - 2048): Maximum length of generated responses
- **Top P** (0.0 - 1.0): Nucleus sampling threshold
- **Top K** (0 - 100): Limits sampling to top K tokens
- **Repetition Penalty** (1.0 - 2.0): Reduces repetitive text

### Vite Configuration

Edit `vite.config.ts` to customize:

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Development server port
    proxy: {
      "/api": {
        target: "http://localhost:8000", // Backend server URL
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
```

## Project Structure

```
ui/
├── public/              # Static assets
├── src/
│   ├── App.tsx         # Main application component
│   ├── main.tsx        # Application entry point
│   └── index.css       # Global styles with Tailwind imports
├── index.html          # HTML template
├── package.json        # Dependencies and scripts
├── tsconfig.json       # TypeScript configuration
├── vite.config.ts      # Vite configuration
├── tailwind.config.js  # Tailwind CSS configuration
└── postcss.config.js   # PostCSS configuration
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Deployment

### Static Hosting

After building, deploy the `dist/` directory to any static hosting service:

- **Netlify**: Drag and drop `dist/` folder
- **Vercel**: `vercel --prod`
- **GitHub Pages**: Push `dist/` to `gh-pages` branch
- **AWS S3**: Sync `dist/` to S3 bucket

### Docker

Example Dockerfile:

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Variables

For production deployments, you may want to use environment variables:

Create `.env.production`:

```env
VITE_API_URL=https://your-backend-api.com
```

Access in code:

```typescript
const serverUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

## Troubleshooting

### Connection Issues

If the UI shows "Disconnected":

1. Verify the backend server is running: `curl http://localhost:8000/health`
2. Check CORS settings in the backend server
3. Update the server URL in Settings panel

### Build Errors

If you encounter TypeScript errors:

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check TypeScript version
npm list typescript
```

### Styling Issues

If Tailwind styles aren't applying:

1. Verify `tailwind.config.js` content paths include your source files
2. Ensure `index.css` imports Tailwind directives
3. Restart the dev server

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## License

Same as the parent project.

## Contributing

1. Make changes in the `src/` directory
2. Test with `npm run dev`
3. Build with `npm run build` to verify production builds
4. Submit a pull request

## Support

For issues specific to the UI, please check:

- Backend server is running and accessible
- CORS is properly configured
- Browser console for error messa
