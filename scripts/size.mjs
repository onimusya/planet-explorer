// Reports raw and gzip (level 9) sizes of the production build in dist/.
// Usage: npm run build && npm run size
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { gzipSync } from 'node:zlib';

const root = 'dist';
const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else files.push(p);
  }
};
try {
  walk(root);
} catch {
  console.error('dist/ not found: run `npm run build` first');
  process.exit(1);
}

const kb = (n) => (n / 1024).toFixed(1).padStart(8) + ' KiB';
let raw = 0, gz = 0;
console.log('file'.padEnd(36) + 'minified'.padStart(12) + 'gzip -9'.padStart(12));
for (const f of files.sort()) {
  const buf = readFileSync(f);
  const g = gzipSync(buf, { level: 9 }).length;
  raw += buf.length;
  gz += g;
  console.log(relative(root, f).replaceAll('\\', '/').padEnd(36) + kb(buf.length) + kb(g));
}
console.log('total'.padEnd(36) + kb(raw) + kb(gz));
console.log(`bytes: ${raw} minified, ${gz} gzip`);
