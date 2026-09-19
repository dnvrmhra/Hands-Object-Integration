import React, { useState, useEffect } from 'react';
import { Rocket, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface CyberTopBarProps {
  hasDeviation: boolean;
}

export const CyberTopBar: React.FC<CyberTopBarProps> = ({ hasDeviation }) => {
  const [met, setMet] = useState(0); // seconds

  useEffect(() => {
    const timer = setInterval(() => setMet(s => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatMET = (s: number): string => {
    const h = Math.floor(s / 3600).toString().padStart(2, '0');
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${h}:${m}:${sec}`;
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 bg-slate-950/80 backdrop-blur-xl border-b border-white/[0.06] flex items-center px-4 gap-4">
      {/* Logo */}
      <div className="flex items-center gap-2 min-w-[180px]">
        <div className="p-1.5 rounded-lg bg-cyan-400/10 border border-cyan-400/20">
          <Rocket className="w-4 h-4 text-cyan-400" />
        </div>
        <div>
          <span className="font-mono font-bold text-cyan-400 tracking-widest text-sm">ASTRA</span>
          <span className="font-mono font-bold text-white/40 text-sm">·</span>
          <span className="font-mono font-bold text-white text-sm">HAR</span>
        </div>
        <div className="h-4 w-px bg-white/10 mx-1" />
        <span className="text-white/30 text-xs font-mono">v1.0.0</span>
      </div>

      {/* Center: MET clock */}
      <div className="flex-1 flex justify-center">
        <div className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-slate-800/60 border border-white/[0.06]">
          <span className="text-white/40 text-xs font-mono">MET</span>
          <span className="text-cyan-400 font-mono font-bold text-lg tracking-widest tabular-nums">
            {formatMET(met)}
          </span>
        </div>
      </div>

      {/* Right: Status pill */}
      <div className="min-w-[180px] flex justify-end">
        <AnimatePresence mode="wait">
          {hasDeviation ? (
            <motion.div
              key="deviation"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-400/10 border border-amber-400/30"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
              <span className="font-mono text-xs font-medium text-amber-400">⚠ DEVIATION DETECTED</span>
            </motion.div>
          ) : (
            <motion.div
              key="nominal"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-400/10 border border-emerald-400/30"
            >
              <div className="relative w-2 h-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400" />
                <div className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-60" />
              </div>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-mono text-xs font-medium text-emerald-400">UPLINK NOMINAL</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
};