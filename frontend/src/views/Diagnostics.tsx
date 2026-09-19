import React from 'react';
import { Gauge, Cpu, Zap, Thermometer, Radio, CheckCircle2 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { SYSTEM_METRICS } from '../data/mockData';

export const Diagnostics: React.FC = () => {
  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">System Health & Edge Diagnostics</h1>
          </div>
          <p className="text-xs text-slate-400">
            Real-time telemetry stream from ISS edge computer: CPU loads, VRAM allocation, core thermals, and tensor FPS.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <CheckCircle2 className="w-4 h-4" /> ALL SUBSYSTEMS NOMINAL
        </div>
      </div>

      {/* Primary KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono">
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>CPU LOAD</span>
            <Cpu className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-sky-400 mt-2">34.2%</div>
          <span className="text-[10px] text-slate-500">8 Cores Active</span>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>GPU VRAM</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-400 mt-2">2.1 GB</div>
          <span className="text-[10px] text-slate-500">Out of 8.0 GB</span>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>CORE TEMP</span>
            <Thermometer className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400 mt-2">64.8°C</div>
          <span className="text-[10px] text-slate-500">Thermal Target: &lt;80°C</span>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>TENSOR FPS</span>
            <Radio className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-2">71.4 FPS</div>
          <span className="text-[10px] text-slate-500">Latency: 14.0 ms</span>
        </div>
      </div>

      {/* Telemetry Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CPU Load Chart */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
              CPU Utilization Profile (%)
            </span>
            <span className="text-[10px] font-mono text-sky-400">120s WINDOW</span>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={SYSTEM_METRICS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickFormatter={t => `${t}s`} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Line type="monotone" dataKey="cpu" stroke="#38bdf8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Inference FPS Chart */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
              YOLOv8 Edge Throughput (FPS)
            </span>
            <span className="text-[10px] font-mono text-emerald-400">NOMINAL &gt; 60 FPS</span>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={SYSTEM_METRICS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickFormatter={t => `${t}s`} />
                <YAxis stroke="#64748b" fontSize={10} domain={[40, 90]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Line type="monotone" dataKey="fps" stroke="#34d399" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Temperature Chart */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
              Workstation Core Thermals (°C)
            </span>
            <span className="text-[10px] font-mono text-amber-400">PASSIVE RADIATOR</span>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={SYSTEM_METRICS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickFormatter={t => `${t}s`} />
                <YAxis stroke="#64748b" fontSize={10} domain={[40, 90]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Line type="monotone" dataKey="temp" stroke="#fbbf24" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* GPU VRAM Chart */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
              GPU Memory Footprint (MB)
            </span>
            <span className="text-[10px] font-mono text-purple-400">UNIFIED MEMORY</span>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={SYSTEM_METRICS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickFormatter={t => `${t}s`} />
                <YAxis stroke="#64748b" fontSize={10} domain={[1000, 3000]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Line type="monotone" dataKey="gpu" stroke="#c084fc" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Diagnostics;
