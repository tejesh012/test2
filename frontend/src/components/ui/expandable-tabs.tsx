"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TabItem {
  title: string;
  Icon: LucideIcon;
  content?: React.ReactNode;
}

interface ExpandableTabsProps {
  tabs: TabItem[];
  onChange?: (index: number | null) => void;
  className?: string;
}

export function ExpandableTabs({ tabs, onChange, className }: ExpandableTabsProps) {
  const [activeTab, setActiveTab] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setActiveTab(null);
        onChange?.(null);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onChange]);

  const handleTabClick = (index: number) => {
    const newValue = activeTab === index ? null : index;
    setActiveTab(newValue);
    onChange?.(newValue);
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "inline-flex items-center gap-1 rounded-full bg-slate-100 p-1.5 dark:bg-slate-800",
        className
      )}
    >
      {tabs.map((tab, index) => {
        const isActive = activeTab === index;
        const Icon = tab.Icon;

        return (
          <motion.button
            key={tab.title}
            onClick={() => handleTabClick(index)}
            className={cn(
              "relative flex items-center gap-2 rounded-full px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white"
                : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
            )}
            layout
          >
            <Icon className="h-4 w-4 shrink-0" />
            <AnimatePresence mode="wait">
              {isActive && (
                <motion.span
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: "auto", opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden whitespace-nowrap"
                >
                  {tab.title}
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        );
      })}
    </div>
  );
}

export default ExpandableTabs;
