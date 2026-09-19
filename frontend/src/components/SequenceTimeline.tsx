import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { ProtocolStep } from '../types/detections';
import { PROTOCOL_STEPS } from '../data/mockData';

interface SequenceTimelineProps {
  activeStep: number;
  onStepClick?: (step: number) => void;
}

const STATUS_COLORS = {
  completed: 'text-emerald-400',
  active:    'text-cyan-400',
  pending:   'text-slate-500',
};

export const SequenceTimeline: React.FC<SequenceTimelineProps> = ({ activeStep, onStepClick }) => {
  const activeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [activeStep]);

  const steps: ProtocolStep[] = PROTOCOL_STEPS.map((s, i) => ({
    ...s,
    status: i < activeStep ? 'completed' : i === activeStep ? 'active' : 'pending',
  }));

  return (
    <div className="flex flex-col gap-1 overflow-y-auto max-h-64 pr-1">
      {steps.map((step, i) => {
        const isActive = i === activeStep;
        const isDone   = i < activeStep;
        return (
          <motion.div
            key={step.id}
            ref={isActive ? activeRef : null}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => onStepClick?.(i)}
            className={`relative flex items-start gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all duration-200 ${
              isActive
                ? 'bg-cyan-400/10 border-l-2 border-cyan-400'
                : isDone
                ? 'bg-emerald-400/5 border-l-2 border-emerald-400/40'
                : 'border-l-2 border-transparent hover:bg-white/[0.03]'
            }`}
          >
            {/* Icon */}
            <div className="mt-0.5 shrink-0">
              {isDone ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : isActive ? (
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}>
                  <Loader2 className="w-4 h-4 text-cyan-400" />
                </motion.div>
              ) : (
                <Circle className="w-4 h-4 text-slate-600" />
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <span className={`text-xs font-mono ${STATUS_COLORS[step.status]}`}>
                    {String(step.id).padStart(2, '0')}
                  </span>
                  <span className={`text-xs font-medium ${isActive ? 'text-white' : isDone ? 'text-slate-300' : 'text-slate-500'}`}>
                    {step.action}
                  </span>
                </div>
                {isDone && step.confidence > 0 && (
                  <span className="text-xs font-mono text-emerald-400 shrink-0">
                    {(step.confidence * 100).toFixed(0)}%
                  </span>
                )}
                {isActive && (
                  <motion.span
                    animate={{ opacity: [1, 0.4, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    className="text-xs font-mono text-cyan-400 shrink-0"
                  >
                    ACTIVE
                  </motion.span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                  step.object === 'GLOVEBOX-01'   ? 'bg-cyan-400/10 text-cyan-400' :
                  step.object === 'CENTRIFUGE-02' ? 'bg-emerald-400/10 text-emerald-400' :
                  step.object === 'SMP-RED'        ? 'bg-rose-500/10 text-rose-500' :
                  'bg-amber-400/10 text-amber-400'
                }`}>
                  {step.object}
                </span>
                {step.timestamp && (
                  <span className="text-xs font-mono text-slate-500">{step.timestamp}</span>
                )}
              </div>
            </div>

            {/* Active pulse */}
            {isActive && (
              <motion.div
                className="absolute inset-0 rounded-lg border border-cyan-400/30"
                animate={{ opacity: [0.3, 0.8, 0.3] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}
          </motion.div>
        );
      })}
    </div>
  );
};