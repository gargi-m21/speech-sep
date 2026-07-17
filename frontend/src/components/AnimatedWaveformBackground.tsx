"use client";

import { useEffect, useRef } from "react";

export default function AnimatedWaveformBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    const waves = [
      { amplitude: 80, frequency: 0.002, speed: 0.02, color: "rgba(168, 85, 247, 0.15)" },   // Purple
      { amplitude: 50, frequency: 0.004, speed: -0.015, color: "rgba(59, 130, 246, 0.12)" }, // Blue
      { amplitude: 100, frequency: 0.001, speed: 0.01, color: "rgba(139, 92, 246, 0.08)" },  // Indigo
    ];

    let phase = 0;

    const animate = () => {
      ctx.clearRect(0, 0, width, height);

      waves.forEach((wave) => {
        ctx.beginPath();
        ctx.strokeStyle = wave.color;
        ctx.lineWidth = 2;

        for (let x = 0; x < width; x++) {
          // Calculate wave height using a sine function and taper it at the edges
          const edgeTaper = Math.sin((x / width) * Math.PI);
          const y =
            height / 2 +
            Math.sin(x * wave.frequency + phase * wave.speed) *
              wave.amplitude *
              edgeTaper;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      });

      phase += 1;
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none opacity-50 z-0" />;
}
