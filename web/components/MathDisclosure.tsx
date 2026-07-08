"use client";

import { useState } from "react";
import { Beaker, ChevronDown } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

interface MathDisclosureProps {
  title?: string;
  children: React.ReactNode;
}

export function MathDisclosure({
  title = "Show the math",
  children,
}: MathDisclosureProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="my-8 overflow-hidden rounded-xl border border-border bg-surface">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="flex w-full items-center justify-between bg-surfaceHover/50 p-4 text-left transition-colors hover:bg-surfaceHover"
      >
        <span className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded border border-border bg-background">
            <Beaker className="h-4 w-4 text-accent" />
          </span>
          <span className="font-mono text-sm font-medium text-primary">
            {title}
          </span>
        </span>
        <motion.span
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="h-5 w-5 text-secondary" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {isOpen ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
          >
            <div className="prose border-t border-border bg-background/50 p-6">
              {children}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
