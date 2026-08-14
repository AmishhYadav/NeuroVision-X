import { defineConfig } from "vitest/config";

// Separate from vite.config.ts on purpose: the app build needs the React and
// Tailwind plugins, the test run needs neither (slicing.ts and render.ts are
// plain TypeScript, no JSX). Keeping them apart means a test-only dependency
// change can never affect `npm run build`.
export default defineConfig({
  test: {
    environment: "node",
    setupFiles: ["./src/test/setup.ts"],
  },
});
