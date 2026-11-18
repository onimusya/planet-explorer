// Keyboard / mouse state. Event handlers only write into preallocated typed arrays.

export const K = {
  W: 0, A: 1, S: 2, D: 3, UP: 4, DOWN: 5, Q: 6, E: 7, SHIFT: 8, ALT: 9,
} as const;

export class Input {
  readonly keys = new Uint8Array(16);
  // accumulated mouse delta since last consume, wheel steps
  readonly mouse = new Float64Array(3);
  // one-shot actions (set by key press, cleared by consumer)
  readonly actions = new Uint8Array(16);
  locked = false;

  constructor(canvas: HTMLCanvasElement) {
    const onKey = (e: KeyboardEvent, down: boolean) => {
      const v = down ? 1 : 0;
      switch (e.code) {
        case 'KeyW': this.keys[K.W] = v; break;
        case 'KeyA': this.keys[K.A] = v; break;
        case 'KeyS': this.keys[K.S] = v; break;
        case 'KeyD': this.keys[K.D] = v; break;
        case 'Space': this.keys[K.UP] = v; break;
        case 'ControlLeft': case 'ControlRight': case 'KeyC': this.keys[K.DOWN] = v; break;
        case 'KeyQ': this.keys[K.Q] = v; break;
        case 'KeyE': this.keys[K.E] = v; break;
        case 'ShiftLeft': case 'ShiftRight': this.keys[K.SHIFT] = v; break;
        case 'AltLeft': case 'AltRight': this.keys[K.ALT] = v; break;
        case 'KeyR': if (down) this.actions[0] = 1; break;
        case 'Digit1': case 'Numpad1': if (down) this.actions[1] = 1; break;
        case 'Digit2': case 'Numpad2': if (down) this.actions[2] = 1; break;
        case 'Digit3': case 'Numpad3': if (down) this.actions[3] = 1; break;
        case 'KeyM': if (down) this.actions[4] = 1; break;
        case 'KeyH': if (down) this.actions[5] = 1; break;
        case 'KeyB': if (down) this.actions[6] = 1; break;
        case 'KeyG': if (down) this.actions[7] = 1; break;
        case 'KeyN': if (down) this.actions[8] = 1; break;
        default: return;
      }
      if (this.locked || e.code === 'Space') e.preventDefault();
    };
    window.addEventListener('keydown', (e) => onKey(e, true));
    window.addEventListener('keyup', (e) => onKey(e, false));
    window.addEventListener('blur', () => this.keys.fill(0));
    canvas.addEventListener('click', () => {
      if (!this.locked) {
        const p = canvas.requestPointerLock({ unadjustedMovement: true } as PointerLockOptions) as unknown as Promise<void> | undefined;
        if (p && typeof p.catch === 'function') p.catch(() => canvas.requestPointerLock());
      }
    });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (!this.locked) this.keys.fill(0);
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.mouse[0] += e.movementX;
      this.mouse[1] += e.movementY;
    });
    canvas.addEventListener('wheel', (e) => {
      this.mouse[2] += e.deltaY > 0 ? -1 : 1;
      e.preventDefault();
    }, { passive: false });
    // Guard against accidental Ctrl+W (Ctrl is the descend key) closing the tab.
    window.addEventListener('beforeunload', (e) => {
      if (this.locked) { e.preventDefault(); }
    });
  }
}
