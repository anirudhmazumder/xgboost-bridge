import { defineConfig } from "tsup";

// The public library, and nothing else: dual ESM + CJS, matching the export map
// in package.json. Tests are plain JavaScript (see D014) and are executed
// directly by `node --test`, so there is no test build step here.
export default defineConfig({
  entry: ["src/index.ts"],
  outDir: "dist",
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  target: "es2020",
  platform: "neutral",
});
