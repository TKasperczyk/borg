import { useEffect, useRef } from "react";

// Particle cloud -- a wide swarm of green wisps across the entire background
// that bends inward toward the active node like light around a gravity well.
//
// Mental model: every particle has a "rest" position uniformly distributed
// across the canvas. When a node is highlighted, each rest position is
// *deformed* toward the focal -- close-to-moderate distances bend the most,
// far distances barely move. The particle is dynamically pulled toward this
// deformed position (springy, with brownian noise so the cloud stays alive).
//
// Geometric effect: the whole field of particles compresses inward, density
// visibly increases around the focal (gravitational lensing magnification),
// and yet particles remain visible across the full background. The cloud
// "leans in" from all sides rather than translating, spotlighting, or
// clustering.
//
//   target  : { x, y } in SVG viewBox coords, or null when idle
//   viewW/H : SVG viewBox dimensions
//   density : particle count (200-400 reads as a full field)
//   enabled : false hides the canvas

export type ParticleTarget = { x: number; y: number } | null;

type Particle = {
  bright: boolean;
  homeXNorm: number;
  homeYNorm: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseR: number;
  baseAlpha: number;
  targetK: number;
  noise: number;
  breathPhase: number;
  breathSpeed: number;
  initialized: boolean;
};

export function ParticleField({
  target,
  viewW = 1200,
  viewH = 660,
  density = 320,
  enabled = true,
}: {
  target: ParticleTarget;
  viewW?: number;
  viewH?: number;
  density?: number;
  enabled?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const targetRef = useRef<ParticleTarget>(target);

  useEffect(() => {
    targetRef.current = target;
    // Track only the focal coordinates; identity of `target` changes each
    // render but the animation only cares about its x/y.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target ? target.x : null, target ? target.y : null]);

  useEffect(() => {
    if (!enabled) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    // Alias to a non-null-typed const so the narrowing survives into the
    // hoisted tick() closure (TS drops control-flow narrowing across it).
    const ctx: CanvasRenderingContext2D = context;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    let W = 0;
    let H = 0;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      W = Math.max(1, rect.width);
      H = Math.max(1, rect.height);
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function svgToCanvas(sx: number, sy: number) {
      const scale = Math.min(W / viewW, H / viewH);
      const cw = viewW * scale;
      const ch = viewH * scale;
      const ox = (W - cw) / 2;
      const oy = (H - ch) / 2;
      return { x: ox + sx * scale, y: oy + sy * scale, scale };
    }

    const count = Math.max(40, Math.floor(density));
    // Particles: random home position on a jittered grid for an even visual
    // distribution (pure random clumps too much at this count).
    const particles: Particle[] = Array.from({ length: count }, (_, i) => {
      const cols = Math.ceil(Math.sqrt(count * 1.78));
      const rows = Math.ceil(count / cols);
      const col = i % cols;
      const row = Math.floor(i / cols) % rows;
      const jx = (Math.random() - 0.5) * 0.9;
      const jy = (Math.random() - 0.5) * 0.9;
      const homeXNorm = (col + 0.5 + jx) / cols;
      const homeYNorm = (row + 0.5 + jy) / rows;

      const bright = i % 4 === 0;
      return {
        bright,
        homeXNorm: Math.max(0.02, Math.min(0.98, homeXNorm)),
        homeYNorm: Math.max(0.02, Math.min(0.98, homeYNorm)),
        x: 0,
        y: 0,
        vx: (Math.random() - 0.5) * 28,
        vy: (Math.random() - 0.5) * 18,
        baseR: bright ? 1.1 + Math.random() * 1.3 : 0.55 + Math.random() * 0.95,
        baseAlpha: bright ? 0.28 + Math.random() * 0.16 : 0.11 + Math.random() * 0.1,
        // Spring toward the deformed target position. Moderate so particles
        // visibly lag and curve as the focal moves.
        targetK: 1.6 + Math.random() * 1.0,
        noise: 30 + Math.random() * 22,
        breathPhase: Math.random() * Math.PI * 2,
        breathSpeed: 0.35 + Math.random() * 0.7,
        initialized: false,
      };
    });

    // Focal state used to deform particle target positions
    let focalX = 0;
    let focalY = 0;
    let focalInited = false;
    let focusStrength = 0;

    let raf = 0;
    let last = performance.now();
    let running = true;

    const onVis = () => {
      running = document.visibilityState === "visible";
      if (running) {
        last = performance.now();
        raf = requestAnimationFrame(tick);
      } else {
        cancelAnimationFrame(raf);
      }
    };
    document.addEventListener("visibilitychange", onVis);

    function tick(now: number) {
      if (!running) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const t = now / 1000;

      if (!focalInited && W > 0 && H > 0) {
        focalX = W / 2;
        focalY = H / 2;
        focalInited = true;
      }

      const tgt = targetRef.current;
      if (tgt) {
        const pp = svgToCanvas(tgt.x, tgt.y);
        // Focal point itself drifts smoothly toward the target node
        const moveK = Math.min(1, dt * 2.4);
        focalX += (pp.x - focalX) * moveK;
        focalY += (pp.y - focalY) * moveK;
        focusStrength += (1 - focusStrength) * Math.min(1, dt * 1.4);
      } else {
        // Gentle sway around canvas-center when nothing is highlighted --
        // the field still "breathes" with no lensing.
        const swayX = Math.sin(t * 0.13) * 30 + Math.sin(t * 0.31) * 10;
        const swayY = Math.cos(t * 0.17) * 20 + Math.sin(t * 0.27) * 7;
        const moveK = Math.min(1, dt * 0.6);
        focalX += (W / 2 + swayX - focalX) * moveK;
        focalY += (H / 2 + swayY - focalY) * moveK;
        focusStrength += (0 - focusStrength) * Math.min(1, dt * 1.2);
      }

      // Lens parameters: sigma controls how wide the "well" is; alphaMax
      // controls how aggressively particles bend inward at the lens peak.
      const sigma = Math.max(W, H) * 0.42;
      const sigma2x2 = sigma * sigma * 2;
      const alphaMax = 0.62; // max fraction of (focal - home) applied

      // Initialize particles at their home positions (no lens at first)
      for (const p of particles) {
        if (!p.initialized) {
          p.x = p.homeXNorm * W;
          p.y = p.homeYNorm * H;
          p.initialized = true;
        }
      }

      // Integrate: spring toward lens-deformed target + brownian noise.
      for (const p of particles) {
        const homeX = p.homeXNorm * W;
        const homeY = p.homeYNorm * H;

        // Vector home -> focal
        const dx = focalX - homeX;
        const dy = focalY - homeY;
        const d2 = dx * dx + dy * dy;

        // Lens weight -- Gaussian falloff with distance. Close-to-moderate
        // distances bend most; far distances barely shift. With dx=dy=0
        // (particle home at focal), weight*(dx,dy)=0 so it stays put.
        const w = Math.exp(-d2 / sigma2x2);
        const lensA = alphaMax * w * focusStrength;

        // Deformed target position
        const targetX = homeX + dx * lensA;
        const targetY = homeY + dy * lensA;

        // Spring + noise + damping
        const ex = targetX - p.x;
        const ey = targetY - p.y;
        const fx = ex * p.targetK;
        const fy = ey * p.targetK;
        const nx = (Math.random() - 0.5) * p.noise;
        const ny = (Math.random() - 0.5) * p.noise;
        p.vx += (fx + nx) * dt;
        p.vy += (fy + ny) * dt;
        p.vx *= 0.92;
        p.vy *= 0.92;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
      }

      ctx.clearRect(0, 0, W, H);

      // Soft body-tint glow centered on focal -- additive depth, very subtle
      if (focusStrength > 0.05) {
        const bodyR = Math.max(W, H) * 0.75;
        const bodyAlpha = 0.05 * focusStrength;
        const grad = ctx.createRadialGradient(focalX, focalY, 0, focalX, focalY, bodyR);
        grad.addColorStop(0, `oklch(0.78 0.13 142 / ${bodyAlpha.toFixed(3)})`);
        grad.addColorStop(0.55, `oklch(0.78 0.13 142 / ${(bodyAlpha * 0.4).toFixed(3)})`);
        grad.addColorStop(1, "oklch(0.78 0.13 142 / 0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);
      }

      const drawTier = (wantBright: boolean) => {
        for (const p of particles) {
          if (p.bright !== wantBright) continue;
          const breath = 0.55 + Math.sin(p.breathPhase + t * p.breathSpeed) * 0.35;
          const alpha = p.baseAlpha * breath;
          if (alpha < 0.005) continue;
          const r = p.baseR;

          const glowR = r * (wantBright ? 4.6 : 3.5);
          const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR);
          grad.addColorStop(0, `oklch(0.84 0.155 142 / ${(alpha * 0.5).toFixed(3)})`);
          grad.addColorStop(1, "oklch(0.84 0.155 142 / 0)");
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
          ctx.fill();

          const coreLight = wantBright ? 0.92 : 0.78;
          ctx.fillStyle = `oklch(${coreLight} 0.13 142 / ${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      drawTier(false);
      drawTier(true);

      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [density, viewW, viewH, enabled]);

  if (!enabled) return null;
  return <canvas ref={canvasRef} className="fc-particles" aria-hidden="true" />;
}
