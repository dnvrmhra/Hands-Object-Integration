import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, XCircle, Zap } from 'lucide-react';
import { PROTOCOL_STEPS } from '../data/mockData';

interface ValidationControlsProps {
  activeStep: number;
  onValidate: () => void;
  onDeviation: () => void;
  onSkip: () => void;
  confidence: number;
}

export const ValidationControls: React.FC<ValidationControlsProps> = ({
  activeStep, onValidate, onDeviation, onSkip, confidence,
}) => {
  const [flash, setFlash] = useState<'validate' | 'deviation' | 'skip' | null>(null);

  const handleClick = (action: 'validate' | 'deviation' | 'skip') => {
    setFlash(action);
    setTimeout(() => setFlash(null), 400);
    if (action === 'validate')  onValidate();
    if (action === 'deviation') onDeviation();
    if (action === 'skip')      onSkip();
  };

  const step = PROTOCOL_STEPS[activeStep];
  const pct = confidence * 100;
  const circumference = 2 * Math.PI * 20;
  const dashoffset = circumference * (1 - confidence);

  return (
    <div className="flex flex-col gap-3">
      {/* Confidence gauge */}
      <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/60 border border-white/[0.06]">
        <div className="relative w-12 h-12 shrink-0">
          <svg className="w-12 h-12 -rotate-90" viewBox="0 0 48 48">
            <circle cx="24" cy="24" r="20" fill="none" stroke="#1e2d3d" strokeWidth="4" />
            <motion.circle
              cx="24" cy="24" r="20" fill="none"
              stroke={pct >= 90 ? '#34d399' : pct >= 75 ? '#fbbf24' : '#f43f5e'}
              strokeWidth="4"
              strokeDasharray={circumference}
              strokeDashoffset={dashoffset}
              strokeLinecap="round"
              animate={{ strokeDashoffset: dashoffset }}
              transition={{ duration: 0.5 }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-mono font-bold text-white">{pct.toFixed(0)}%</span>
          </div>
        </div>
        <div className="flex-1">
          <div className="text-xs text-slate-400 font-mono">STEP {activeStep + 1} CONFIDENCE</div>
          <div className="text-sm font-medium text-white truncate">{step?.action ?? '—'}</div>
          <div className="text-xs font-mono text-slate-500 truncate">{step?.object ?? '—'}</div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="grid grid-cols-3 gap-2">
        {/* Validate */}
        <motion.button
          animate={flash === 'validate' ? { scale: [1, 1.08, 1], backgroundColor: ['#34d39920', '#34d39950', '#34d39920'] } : {}}
          onClick={() => handleClick('validate')}
          className="relative flex flex-col items-center gap-1 py-3 rounded-lg bg-emerald-400/10 border border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/20 transition-all"
        >
          <CheckCircle2 className="w-5 h-5" />
          <span className="text-xs font-mono font-medium">VALIDATE</span>
          <AnimatePresence>
            {flash === 'validate' && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-lg bg-emerald-400/20" />
            )}
          </AnimatePresence>
        </motion.button>

        {/* Deviation */}
        <motion.button
          animate={flash === 'deviation' ? { scale: [1, 1.08, 1] } : {}}
          onClick={() => handleClick('deviation')}
          className="relative flex flex-col items-center gap-1 py-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-500 hover:bg-rose-500/20 transition-all"
        >
          <XCircle className="w-5 h-5" />
          <span className="text-xs font-mono font-medium">DEVIATION</span>
          <AnimatePresence>
            {flash === 'deviation' && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-lg bg-rose-500/20" />
            )}
          </AnimatePresence>
        </motion.button>

        {/* Skip */}
        <motion.button
          animate={flash === 'skip' ? { scale: [1, 1.08, 1] } : {}}
          onClick={() => handleClick('skip')}
          className="relative flex flex-col items-center gap-1 py-3 rounded-lg bg-amber-400/10 border border-amber-400/30 text-amber-400 hover:bg-amber-400/20 transition-all"
        >
          <Zap className="w-5 h-5" />
          <span className="text-xs font-mono font-medium">SKIP</span>
          <AnimatePresence>
            {flash === 'skip' && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-lg bg-amber-400/20" />
            )}
          </AnimatePresence>
        </motion.button>
      </div>

      {/* Guidance text */}
      {step && (
        <div className="p-2.5 rounded-lg bg-slate-800/40 border border-white/[0.04]">
          <div className="text-xs text-slate-400 font-mono mb-1">GUIDANCE</div>
          <p className="text-xs text-slate-300 leading-relaxed">{step.guidance}</p>
        </div>
      )}
    </div>
  );
};