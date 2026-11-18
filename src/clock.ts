// Per-frame scalars shared by the frame loop and the systems it drives.
// V8 boxes a fractional number into a new heap object when it is passed to or returned from a
// call that the JIT did not inline, or stored in a closure variable. Per-frame code therefore
// exchanges such values through typed arrays (this one, or a module's own) and object fields.
export const clock = new Float64Array(4);
export const DT = 0; // simulation time step (s), clamped to 0..0.1
export const TIME = 1; // animation time (s): the requestAnimationFrame timestamp
export const FRAME_MS = 2; // time since the previous frame (ms)
export const CPU_MS = 3; // CPU time of the frame, from callback start to queue submit (ms)
