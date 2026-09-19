import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingDown, Wifi } from 'lucide-react';

export const BandwidthPanel: React.FC = () => {
  const [savings, setSavings] = useState(0);
  const [downlink, setDownlink] = useState(0);

  useEffect(() => {
    // Count up savings
    let s = 0;
    const countup = setInterval(() => {
      s = Math.min(s + 2.5, 99.2);
      setSavings(s);
      if (s >= 99.2) clearInterval(countup);
    }, 30);

    // Live downlink counter
    const dl = setInterval(() => {
      setDownlink(prev => prev + Math.floor(Math.random() * 512 + 256));
    }, 200);

    return () => { clearInterval(countup); clearInterval(dl); };
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <TrendingDown className="w-3 h-3 text-emerald-400" /> Bandwidth
        </div>
        <div className="flex items-center gap-1.5">
          <Wifi className="w-3 h-3 text-cyan-400" />
          <span className="text-xs font-mono text-cyan-400">{(downlink / 1024).toFixed(1)} KB/s</span>
        </div>
      </div>

      {/* Raw bar */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-xs font-mono text-rose-500">RAW VIDEO</span>
          <span className="text-xs font-mono text-slate-400">2.4 GB/hr</span>
        </div>
        <div className="h-3 w-full rounded-full bg-rose-500/20 overflow-hidden border border-rose-500/20">
          <div className="h-full w-full bg-rose-500/60 rounded-full" />
        </div>
      </div>

      {/* Compressed bar */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-xs font-mono text-emerald-400">ASTRA-HAR</span>
          <span className="text-xs font-mono text-slate-400">18 MB/hr</span>
        </div>
        <div className="h-3 w-full rounded-full bg-slate-700 overflow-hidden border border-emerald-400/20">
          <motion.div
            className="h-full bg-emerald-400/70 rounded-full"
            style={{ width: `${(18 / 2457) * 100}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${(18 / 2457) * 100}%` }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Big savings number */}
      <div className="flex items-center justify-center py-1">
        <span className="text-xs font-mono text-slate-400 mr-2">SAVINGS</span>
        <motion.span
          className="text-2xl font-mono font-bold text-emerald-400"
          animate={{ opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          {savings.toFixed(1)}%
        </motion.span>
      </div>
    </div>
  );
};