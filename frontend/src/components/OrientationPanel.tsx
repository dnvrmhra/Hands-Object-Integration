import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Attitude3D } from '../types/detections';
import { Compass, ShieldCheck } from 'lucide-react';

interface OrientationPanelProps {
  attitude?: Attitude3D;
}

export const OrientationPanel: React.FC<OrientationPanelProps> = ({ attitude }) => {
  const [simulatedAngles, setSimulatedAngles] = useState({ roll: 0, pitch: 0, yaw: 0 });

  useEffect(() => {
    if (attitude) return; // If real attitude is provided, do not run simulation
    let t = 0;
    const interval = setInterval(() => {
      t += 0.05;
      setSimulatedAngles({
        roll: Math.sin(t * 0.7) * 5,
        pitch: Math.sin(t * 0.5) * 3,
        yaw: (t * 0.3) % 360,
      });
    }, 50);
    return () => clearInterval(interval);
  }, [attitude]);

  const roll = attitude ? attitude.roll : simulatedAngles.roll;
  const pitch = attitude ? attitude.pitch : simulatedAngles.pitch;
  const yaw = attitude ? attitude.yaw : (simulatedAngles.yaw % 180) - 90;

  const gauges = [
    { label: 'ROLL (\u03c6)', value: roll, limit: 180, unit: '\u00b0' },
    { label: 'PITCH (\u03b8)', value: pitch, limit: 90, unit: '\u00b0' },
    { label: 'YAW (\u03c8)', value: yaw, limit: 180, unit: '\u00b0' },
  ];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-mono text-slate-300 uppercase tracking-wider flex items-center gap-1.5 font-bold">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          3D Attitude (6-DOF Tracking)
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-400/20 flex items-center gap-1">
          <ShieldCheck className="w-3 h-3" />
          ORIENTATION-AGNOSTIC
        </span>
      </div>

      <div className="flex gap-2.5">
        {gauges.map(g => {
          const pct = Math.min(1.0, Math.abs(g.value) / g.limit);
          const color = pct > 0.7 ? '#f43f5e' : pct > 0.4 ? '#fbbf24' : '#38bdf8';
          const barW = Math.min(100, Math.max(8, pct * 100));

          return (
            <div
              key={g.label}
              className="flex-1 flex flex-col items-center gap-1.5 p-2 rounded-lg bg-slate-800/60 border border-white/[0.06]"
            >
              <span className="text-[10px] font-mono text-slate-400">{g.label}</span>
              <motion.span
                className="text-base font-mono font-bold tabular-nums"
                style={{ color }}
                animate={{ color }}
                transition={{ duration: 0.2 }}
              >
                {g.value >= 0 ? '+' : ''}
                {g.value.toFixed(1)}
                {g.unit}
              </motion.span>
              {/* Visual gauge bar */}
              <div className="w-full h-1.5 rounded-full bg-slate-700/80 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    backgroundColor: color,
                    width: `${barW}%`,
                    marginLeft: g.value < 0 ? `${Math.max(0, 50 - barW / 2)}%` : '50%',
                  }}
                  animate={{ width: `${barW}%`, backgroundColor: color }}
                  transition={{ duration: 0.2 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};