import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

// Loads .wgsl files as strings. Supports `//#include "file.wgsl"` (include-once per module)
// and minifies (comment/whitespace stripping) for production builds.
function wgsl(): Plugin {
  let isBuild = false;
  return {
    name: 'wgsl-loader',
    configResolved(c) {
      isBuild = c.command === 'build';
    },
    load(id) {
      const clean = id.split('?')[0];
      if (!clean.endsWith('.wgsl')) return null;
      const seen = new Set<string>();
      const deps: string[] = [];
      const expand = (file: string): string => {
        const abs = path.resolve(file);
        if (seen.has(abs)) return '';
        seen.add(abs);
        deps.push(abs);
        const src = fs.readFileSync(abs, 'utf8');
        return src.replace(/^[ \t]*\/\/#include\s+"([^"]+)"[ \t]*$/gm, (_m, rel: string) =>
          expand(path.resolve(path.dirname(abs), rel)),
        );
      };
      let code = expand(clean);
      for (const d of deps) this.addWatchFile(d);
      if (isBuild) code = minifyWgsl(code);
      return `export default ${JSON.stringify(code)};`;
    },
  };
}

const MULTI_OPS = ['>>=', '<<=', '->', '>=', '<=', '==', '!=', '&&', '||', '<<', '>>', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^='];
const OPCH = '+-*/%<>=!&|^';

function minifyWgsl(src: string): string {
  src = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
  const re = /[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+[iu]?|(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?[fhiu]?|>>=|<<=|->|>=|<=|==|!=|&&|\|\||<<|>>|\+\+|--|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|\S/g;
  const tokens = src.match(re) ?? [];
  let out = '';
  let prev = '';
  const isWord = (c: string) => /[A-Za-z0-9_]/.test(c);
  for (const t of tokens) {
    if (prev) {
      const a = prev[prev.length - 1];
      const b = t[0];
      if ((isWord(a) && isWord(b)) || (OPCH.includes(a) && OPCH.includes(b))) out += ' ';
    }
    out += t;
    prev = t;
  }
  void MULTI_OPS;
  return out;
}

export default defineConfig({
  plugins: [wgsl()],
  build: {
    target: 'es2022',
    modulePreload: false,
    reportCompressedSize: true,
    assetsInlineLimit: 0,
  },
});
