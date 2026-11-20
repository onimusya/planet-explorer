// Worker: builds the collision distance fields (sdf.ts) off the main thread. Receives
// { key, P, tris, res, closeR } jobs and answers each with { key, sdf } (sdf null: no solid
// triangles), transferring the arrays.
import { buildSdf } from './sdf';

interface Job { key: number; P: Float32Array; tris: Uint32Array; res: number; closeR: number }

self.onmessage = (e: MessageEvent<Job[]>) => {
  for (const j of e.data) {
    const sdf = buildSdf(j.P, j.tris, j.res, j.closeR);
    (self as unknown as Worker).postMessage({ key: j.key, sdf }, sdf ? [sdf.d.buffer] : []);
  }
};
