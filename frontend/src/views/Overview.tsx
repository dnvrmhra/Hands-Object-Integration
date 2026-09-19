import React from 'react';
import { ShieldCheck, Clock, CheckCircle2, Zap, ArrowRight, Radio, Layers, Eye } from 'lucide-react';
import SpinningGlobe from '../components/SpinningGlobe';

interface OverviewProps {
  onNavigate: (view: string) => void;
  activeStep: number;
}

export const Overview: React.FC<OverviewProps> = ({ onNavigate, activeStep }) => {
  const pipelineStages = [
    { name: 'CAMERA INGEST', latency: '2.1 ms', status: 'active', desc: '1080p60 ISS Optical Stream' },
    { name: 'EDGE PREPROCESS', latency: '4.3 ms', status: 'active', desc: 'RGB Normalization & Letterbox' },
    { name: 'YOLOv8 DETECT', latency: '14.2 ms', status: 'active', desc: '4 Microgravity Classes' },
    { name: 'POSE & HOI', latency: '6.8 ms', status: 'active', desc: '21-pt Mesh & Vector Ray' },
    { name: 'SEQ VALIDATOR', latency: '1.2 ms', status: 'active', desc: 'FSM Microgravity Rules' },
    { name: 'DOWNLINK BUS', latency: '0.9 ms', status: 'active', desc: 'JSON Event Payload (18 MB/h)' },
  ];

  return (
    <div className="space-y-6 pb-8">
      {/* Top Banner with 3D Globe */}
      <div className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-7 space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-sky-400/30 bg-sky-500/10 text-sky-400 text-xs font-mono uppercase tracking-widest">
              <Radio className="w-3.5 h-3.5 animate-pulse" />
              ISRO PS-26174 • Columbus Payload Telemetry
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl font-sans">
              ASTRA-HAR Vision Hub
            </h1>
            <p className="text-slate-400 text-sm leading-relaxed max-w-xl">
              Autonomous microgravity activity recognition powered by edge-optimized Ultralytics YOLOv8. 
              Real-time biology protocol monitoring inside the Columbus Workstation with sub-15ms edge inference, 
              zero ground-link dependency, and 99.2% telemetry bandwidth reduction.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
              <button
                onClick={() => onNavigate('live')}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-500 text-slate-950 font-semibold text-sm hover:bg-sky-400 transition-colors shadow-lg shadow-sky-500/20"
              >
                <Eye className="w-4 h-4" />
                Launch Live Experiment Deck
              </button>
              <button
                onClick={() => onNavigate('vision')}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-white/10 text-white font-medium text-sm transition-colors"
              >
                <Layers className="w-4 h-4" />
                Inspect AI Neural Tree
              </button>
            </div>
          </div>
          <div className="lg:col-span-5 h-64 relative rounded-xl overflow-hidden border border-white/[0.06] bg-slate-950/60">
            <SpinningGlobe />
            <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-md bg-slate-950/80 border border-white/10 text-[10px] font-mono text-slate-400">
              ISS ORBIT: 418 KM • INCL: 51.6°
            </div>
          </div>
        </div>
      </div>

      {/* Mission KPI Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>PROTOCOL COMPLIANCE</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold font-mono text-emerald-400">94.2%</div>
            <div className="text-xs text-slate-400 mt-1">Nominal • 0 critical violations</div>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div className="bg-emerald-400 h-full rounded-full" style={{ width: '94.2%' }} />
          </div>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>EDGE LATENCY (YOLO)</span>
            <Zap className="w-4 h-4 text-sky-400" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold font-mono text-sky-400">14.2 ms</div>
            <div className="text-xs text-slate-400 mt-1">70.4 FPS • INT8 Quantized</div>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div className="bg-sky-400 h-full rounded-full" style={{ width: '70%' }} />
          </div>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>ACTIVE STEP INDEX</span>
            <CheckCircle2 className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold font-mono text-amber-400">0{activeStep + 1} / 07</div>
            <div className="text-xs text-slate-400 mt-1">Centrifuge Reagent Protocol</div>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div className="bg-amber-400 h-full rounded-full" style={{ width: `${((activeStep + 1) / 7) * 100}%` }} />
          </div>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>DOWNLINK REDUCTION</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold font-mono text-purple-400">99.2%</div>
            <div className="text-xs text-slate-400 mt-1">2.4 GB/h down to 18 MB/h</div>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div className="bg-purple-400 h-full rounded-full" style={{ width: '99.2%' }} />
          </div>
        </div>
      </div>

      {/* 6-Stage AI Decision Ribbon */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-mono text-slate-300 font-semibold tracking-wider uppercase">
            6-Stage Vision & Activity Decision Ribbon
          </h2>
          <span className="text-[11px] font-mono text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-400/20">
            TOTAL INGEST-TO-DOWNLINK: 29.5ms
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {pipelineStages.map((stage, idx) => (
            <div
              key={stage.name}
              className="relative p-3 rounded-lg border border-white/[0.06] bg-slate-950/50 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-mono text-slate-500">STAGE 0{idx + 1}</span>
                  <span className="text-[10px] font-mono text-sky-400">{stage.latency}</span>
                </div>
                <div className="text-xs font-bold font-mono text-slate-200">{stage.name}</div>
                <p className="text-[11px] text-slate-400 mt-1 leading-snug">{stage.desc}</p>
              </div>
              <div className="mt-3 flex items-center justify-between pt-2 border-t border-white/[0.04]">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[9px] font-mono text-emerald-400 uppercase">SYNCHRONIZED</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Spec & Synthetic Dataset Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <h3 className="text-sm font-mono text-slate-200 font-bold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-sky-400" />
            SYNTHETIC DATASET TOPOLOGY
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Procedurally rendered 640×640 Columbus glovebox scene generator simulates ambient lighting variations,
            metal rack specularities, Gaussian micro-vibrations, and ISS motion blur artifacts.
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-2">
            <div className="bg-slate-950/60 p-2 rounded border border-white/5">
              <span className="text-slate-500 block text-[10px]">TOTAL IMAGES</span>
              <span className="text-sky-300 font-bold text-sm">1,000 (800 / 200)</span>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-white/5">
              <span className="text-slate-500 block text-[10px]">TARGET CLASSES</span>
              <span className="text-sky-300 font-bold text-sm">4 Classes (YOLOv8)</span>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-3">
          <h3 className="text-sm font-mono text-slate-200 font-bold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            DEPLOYMENT HARDWARE PROFILE
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Zero-cloud, air-gapped station architecture complying with ISS Flight Rules. ONNX Runtime WebGL/NPU 
            fallback guarantees operational integrity during line-of-sight satellite blackouts.
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-2">
            <div className="bg-slate-950/60 p-2 rounded border border-white/5">
              <span className="text-slate-500 block text-[10px]">NPU LATENCY</span>
              <span className="text-emerald-300 font-bold text-sm">14.2 ms / frame</span>
            </div>
            <div className="bg-slate-950/60 p-2 rounded border border-white/5">
              <span className="text-slate-500 block text-[10px]">DOWNLINK MODE</span>
              <span className="text-emerald-300 font-bold text-sm">JSON Event Bus</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Overview;
