"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, X, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface MenuOption {
  label: string;
  onClick: () => void;
  Icon: LucideIcon;
}

interface FloatingActionMenuProps {
  options: MenuOption[];
  className?: string;
}

export function FloatingActionMenu({ options, className }: FloatingActionMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  const menuVariants = {
    closed: {
      transition: { staggerChildren: 0.05, staggerDirection: -1 },
    },
    open: {
      transition: { staggerChildren: 0.07, delayChildren: 0.2 },
    },
  };

  const itemVariants = {
    closed: { opacity: 0, y: 20, scale: 0.8 },
    open: { opacity: 1, y: 0, scale: 1 },
  };

  return (
    <div className={cn("fixed bottom-6 right-6 z-50", className)}>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute bottom-16 right-0 flex flex-col gap-3"
            initial="closed"
            animate="open"
            exit="closed"
            variants={menuVariants}
          >
            {options.map((option, index) => (
              <motion.div
                key={option.label}
                variants={itemVariants}
                className="flex items-center gap-3 justify-end"
              >
                <span className="bg-slate-900 text-white px-3 py-1.5 rounded-lg text-sm whitespace-nowrap shadow-lg">
                  {option.label}
                </span>
                <Button
                  variant="secondary"
                  size="icon"
                  onClick={() => {
                    option.onClick();
                    setIsOpen(false);
                  }}
                  className="h-12 w-12 rounded-full shadow-lg hover:shadow-xl transition-shadow bg-white dark:bg-slate-800"
                >
                  <option.Icon className="h-5 w-5" />
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        animate={{ rotate: isOpen ? 45 : 0 }}
        transition={{ duration: 0.2 }}
      >
        <Button
          onClick={toggleMenu}
          size="icon"
          className="h-14 w-14 rounded-full shadow-xl hover:shadow-2xl transition-all"
        >
          {isOpen ? <X className="h-6 w-6" /> : <Plus className="h-6 w-6" />}
        </Button>
      </motion.div>
    </div>
  );
}

export default FloatingActionMenu;
