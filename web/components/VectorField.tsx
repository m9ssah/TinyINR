"use client";

import { useEffect, useRef } from "react";

export function VectorField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    let animationFrameId = 0;
    let time = 0;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;

      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const draw = () => {
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;

      context.clearRect(0, 0, width, height);

      const spacing = 40;
      const cols = Math.floor(width / spacing) + 1;
      const rows = Math.floor(height / spacing) + 1;

      for (let i = 0; i < cols; i += 1) {
        for (let j = 0; j < rows; j += 1) {
          const x = i * spacing;
          const y = j * spacing;
          const angle =
            Math.sin(x * 0.01 + time) * Math.cos(y * 0.01 + time) * Math.PI;
          const length = 12 + Math.sin(x * 0.02 - time) * 4;
          const dx = x - width / 2;
          const dy = y - height / 2;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxDist = Math.max(width, height) / 2;
          const opacity = Math.max(0.1, 1 - dist / maxDist);

          context.save();
          context.translate(x, y);
          context.rotate(angle);

          context.beginPath();
          context.moveTo(0, 0);
          context.lineTo(length, 0);
          context.strokeStyle = `rgba(57, 211, 187, ${opacity * 0.5})`;
          context.lineWidth = 1.5;
          context.stroke();

          context.beginPath();
          context.moveTo(length, 0);
          context.lineTo(length - 4, -3);
          context.lineTo(length - 4, 3);
          context.fillStyle = `rgba(57, 211, 187, ${opacity * 0.8})`;
          context.fill();

          context.restore();
        }
      }

      time += 0.01;
      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener("resize", resize);
    resize();
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full opacity-40"
      style={{ mixBlendMode: "screen" }}
    />
  );
}
