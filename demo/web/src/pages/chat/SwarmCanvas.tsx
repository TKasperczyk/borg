import { useEffect, useRef } from "react";

import type { TurnPhaseName } from "../../api/types";
import {
  breathForArousal,
  swarmParamsForPhase,
  type DelibPath,
  type SwarmOutcome,
} from "./swarmCore";

type Particle = {
  ang: number;
  r: number;
  cr: number;
  sp: number;
  shell: number;
};

type FlyingSquare = {
  ang: number;
  t: number;
  label: string;
};

type TokenDash = {
  ang: number;
  t: number;
  sp: number;
};

export type SwarmCanvasProps = {
  phase: TurnPhaseName | null;
  delibPath: DelibPath;
  outcome: SwarmOutcome;
  arousal: number;
  hue: number;
  inFlight: boolean;
  evidencePulse: number;
  tokenPulse: number;
};

function color(hue: number, alpha: number, light = 0.78, chroma = 0.13): string {
  return `oklch(${light} ${chroma} ${hue} / ${alpha})`;
}

function initParticles(): Particle[] {
  return Array.from({ length: 110 }, (_, index) => ({
    ang: Math.random() * Math.PI * 2,
    r: 60 + Math.random() * 50,
    cr: 70 + Math.random() * 35,
    sp: 0.0003 + Math.random() * 0.0007,
    shell: index % 2,
  }));
}

function drawCoreBlob(ctx: CanvasRenderingContext2D, hue: number, r: number): void {
  const gradient = ctx.createRadialGradient(0, 0, r * 0.1, 0, 0, r * 1.9);
  gradient.addColorStop(0, color(hue, 0.85, 0.82));
  gradient.addColorStop(0.45, color(hue, 0.3));
  gradient.addColorStop(1, color(hue, 0));
  ctx.beginPath();
  ctx.arc(0, 0, r * 1.9, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = color(hue, 0.9);
  ctx.stroke();
}

function drawOutcomeAfterglow(
  ctx: CanvasRenderingContext2D,
  hue: number,
  outcome: SwarmOutcome,
  sinceOutcomeSeconds: number,
  now: number,
): void {
  if (outcome === "idle" || sinceOutcomeSeconds > 6) {
    return;
  }

  const fade = Math.max(0, 1 - sinceOutcomeSeconds / 6);
  if (outcome === "silence") {
    ctx.beginPath();
    ctx.arc(0, 0, 108, 0, Math.PI * 2);
    ctx.lineWidth = 1;
    ctx.strokeStyle = color(hue, 0.45 * fade);
    ctx.stroke();
  } else if (outcome === "observed") {
    const ang = now * 0.0011;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(ang) * 108, Math.sin(ang) * 108);
    ctx.strokeStyle = color(hue, 0.35 * fade);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(0, 0, 108, ang - 0.6, ang);
    ctx.strokeStyle = color(hue, 0.18 * fade);
    ctx.lineWidth = 10;
    ctx.stroke();
  } else if (outcome === "emitted" && sinceOutcomeSeconds < 1.6) {
    const t = sinceOutcomeSeconds / 1.6;
    ctx.beginPath();
    ctx.arc(0, 0, 30 + t * 138, 0, Math.PI * 2);
    ctx.lineWidth = 2 * (1 - t);
    ctx.strokeStyle = color(hue, 0.6 * (1 - t));
    ctx.stroke();
  } else if (outcome === "error") {
    ctx.beginPath();
    ctx.arc(0, 0, 22 + Math.sin(now * 0.05) * 2, 0, Math.PI * 2);
    ctx.strokeStyle = "oklch(0.62 0.19 25 / 0.8)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

export function SwarmCanvas(props: SwarmCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[] | null>(null);
  const evidenceRef = useRef<FlyingSquare[]>([]);
  const tokensRef = useRef<TokenDash[]>([]);
  const previousEvidencePulse = useRef(props.evidencePulse);
  const previousTokenPulse = useRef(props.tokenPulse);
  const outcomeAtRef = useRef(performance.now());
  const previousOutcomeRef = useRef(props.outcome);

  useEffect(() => {
    if (props.evidencePulse !== previousEvidencePulse.current) {
      previousEvidencePulse.current = props.evidencePulse;
      for (let index = 0; index < 14; index += 1) {
        evidenceRef.current.push({
          ang: Math.random() * Math.PI * 2,
          t: -index * 0.045,
          label: "E",
        });
      }
    }
  }, [props.evidencePulse]);

  useEffect(() => {
    if (props.tokenPulse !== previousTokenPulse.current) {
      previousTokenPulse.current = props.tokenPulse;
      tokensRef.current.push({
        ang: Math.PI + (Math.random() - 0.5) * 0.9,
        t: 0,
        sp: 1.6 + Math.random(),
      });
    }
  }, [props.tokenPulse]);

  useEffect(() => {
    if (props.outcome !== previousOutcomeRef.current) {
      previousOutcomeRef.current = props.outcome;
      outcomeAtRef.current = performance.now();
    }
  }, [props.outcome]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return undefined;
    }

    let context: CanvasRenderingContext2D | null = null;
    try {
      context = canvas.getContext("2d");
    } catch {
      context = null;
    }
    if (context === null) {
      return undefined;
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let idleTimer = 0;
    let hidden = document.hidden;

    const cancelScheduled = () => {
      if (raf !== 0) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
      if (idleTimer !== 0) {
        window.clearTimeout(idleTimer);
        idleTimer = 0;
      }
    };

    const schedule = (fullRate: boolean) => {
      if (reducedMotion || hidden) {
        return;
      }

      if (fullRate) {
        raf = window.requestAnimationFrame(draw);
        return;
      }

      idleTimer = window.setTimeout(() => {
        idleTimer = 0;
        if (!hidden) {
          raf = window.requestAnimationFrame(draw);
        }
      }, 83);
    };

    const draw = (now: number) => {
      const afterglowActive = props.outcome !== "idle" && now - outcomeAtRef.current <= 6_000;
      const fullRate = props.inFlight || afterglowActive;
      particlesRef.current ??= initParticles();
      const params = swarmParamsForPhase({
        phase: props.phase,
        delibPath: props.delibPath,
        outcome: props.outcome,
        arousal: props.arousal,
      });
      const breath = breathForArousal(
        now,
        props.arousal,
        props.delibPath === "system_2" && props.phase === "delib" ? 1.8 : 1,
      );

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.save();
      context.scale(2, 2);
      context.translate(164, 122);

      const particleFill = color(props.hue, params.alpha);
      const evidenceFill = color(props.hue, 0.85);
      const tokenFill = color(props.hue, 0.9);

      for (const particle of particlesRef.current) {
        particle.cr += (params.targetR + (particle.r - 90) * 0.5 - particle.cr) * 0.03;
        const dir = params.counterRotate && particle.shell === 1 ? -1 : 1;
        particle.ang += particle.sp * params.speed * dir * 16;
        const jitter = params.jitter === 0 ? 0 : Math.sin(now * 0.01 + particle.ang * 7) * params.jitter;
        const r = particle.cr + jitter + breath * 0.6;
        context.beginPath();
        context.arc(Math.cos(particle.ang) * r, Math.sin(particle.ang) * r, 1.1, 0, Math.PI * 2);
        context.fillStyle = particleFill;
        context.fill();
      }

      drawCoreBlob(context, props.hue, 9 + breath * 0.4);

      evidenceRef.current = evidenceRef.current.filter((item) => item.t < 1.15);
      for (const item of evidenceRef.current) {
        item.t += 0.016;
        if (item.t < 0) {
          continue;
        }
        const t = Math.min(1, item.t);
        const ease = 1 - (1 - t) ** 3;
        const r = 105 * (1 - ease * 0.78);
        const x = Math.cos(item.ang) * r;
        const y = Math.sin(item.ang) * r;
        context.globalAlpha = t > 0.95 ? (1.15 - item.t) * 5 : 0.9;
        context.fillStyle = evidenceFill;
        context.fillRect(x - 3, y - 3, 6, 6);
        context.fillStyle = "#0B0B09";
        context.font = "700 5px JetBrains Mono";
        context.textAlign = "center";
        context.fillText(item.label, x, y + 2);
        context.globalAlpha = 1;
      }

      tokensRef.current = tokensRef.current.filter((item) => item.t < 70);
      for (const item of tokensRef.current) {
        item.t += 1;
        const barrierR = 92;
        let d = 12 + item.t * item.sp;
        let alpha = Math.max(0, 1 - item.t / 60);
        if (props.outcome === "suppressed" && d > barrierR) {
          d = barrierR - (d - barrierR) * 0.6;
          alpha *= 0.5;
        }
        context.save();
        context.translate(Math.cos(item.ang) * d, Math.sin(item.ang) * d);
        context.rotate(item.ang);
        context.globalAlpha = alpha;
        context.fillStyle = tokenFill;
        context.fillRect(-3.5, -0.8, 7, 1.6);
        context.restore();
        context.globalAlpha = 1;
      }

      if (params.barrier || props.outcome === "suppressed") {
        context.save();
        context.setLineDash([4, 4]);
        context.beginPath();
        context.arc(0, 0, 92, Math.PI * 0.55, Math.PI * 1.45);
        context.lineWidth = 2;
        context.strokeStyle = "oklch(0.62 0.19 25 / 0.75)";
        context.stroke();
        context.restore();
      }

      drawOutcomeAfterglow(
        context,
        props.hue,
        props.outcome,
        (now - outcomeAtRef.current) / 1000,
        now,
      );

      context.restore();

      schedule(fullRate);
    };

    const onVisibility = () => {
      hidden = document.hidden;
      if (!hidden && !reducedMotion) {
        cancelScheduled();
        raf = window.requestAnimationFrame(draw);
      }
    };

    document.addEventListener("visibilitychange", onVisibility);
    raf = window.requestAnimationFrame(draw);

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      cancelScheduled();
    };
  }, [
    props.arousal,
    props.delibPath,
    props.hue,
    props.inFlight,
    props.outcome,
    props.phase,
  ]);

  return <canvas ref={canvasRef} width={656} height={480} className="mind-canvas" />;
}
