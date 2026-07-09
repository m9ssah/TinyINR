"use client";

import { motion } from "framer-motion";

interface MotionRevealProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}

export function MotionReveal({
  children,
  className,
  delay = 0,
}: MotionRevealProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
