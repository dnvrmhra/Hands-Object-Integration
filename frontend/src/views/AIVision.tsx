import React from 'react';
import { Network, Database, Cpu, Activity, Zap, CheckCircle2 } from 'lucide-react';
import { useMockDetections } from '../hooks/useMockDetections';

interface AIVisionProps {
  activeStep: number;
}

export const AIVision: React.FC<AIVisionProps> = ({ activeStep }) => {
  const { frame } = useMockDetections(activeStep);

  const dagNodes = [
    { id: 'n1', title: 'Input Frame', desc: '1080p RGB → Letterbox 640×640×3', shape: '[1, 3, 640, 640]', type: 'input' },
    { id: 'n2', title: 'Backbone CSPDarknet', desc: 'C2f cross-stage partial features', shape: '[1, 256, 80, 80]', type: 'conv' },
    { id: 'n3', title: 'PANet Neck', desc: 'Multi-scale spatial feature aggregation', shape: '[P3, P4, P5 Pyramid]', type: 'neck' },
    { id: 'n4', title: 'Decoupled Detect Head', desc: 'Anchor-free bbox & class prediction', shape: '[1, 8, 8400] Tensors', type: 'head' },
    { id: 'n5', title: 'NMS Vector Filter', desc: 'IoU > 0.45 Supression & Score > 0.60', shape: 'Top-K Projections', type: 'post' },
    { id: 'n6', title: 'FSM Semantic Engine', desc: 'Spatial HOI & Microgravity validation', shape: 'JSON Protocol Events', type: 'output' },
  ];

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">AI Vision Tree & Neural Inspection</h1>
          </div>
          <p className="text-xs text-slate-400">
            Directed Acyclic Graph (DAG) inspection of the edge YOLOv8 tensor pipeline and real-time entity state vectors.
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <span className="px-3 py-1 rounded bg-sky-500/10 border border-sky-400/20 text-sky-400">
            MODEL: YOLOv8n-MICROGRAVITY
          </span>
          <span className="px-3 py-1 rounded bg-emerald-500/10 border border-emerald-400/20 text-emerald-400">
            STATUS: 71.2 FPS NOMINAL
          </span>
        </div>
      </div>

      {/* Main Grid: DAG (Left), Entity Ledger (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Neural DAG Visualization (5 cols) */}
        <div className="lg:col-span-5 rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-sky-400" /> Neural Processing DAG
            </span>
            <span className="text-[10px] font-mono text-slate-500">6 CONNECTED NODES</span>
          </div>

          <div className="space-y-3 relative">
            {dagNodes.map((node, i) => (
              <div key={node.id} className="relative">
                <div className="p-3 rounded-lg border border-white/[0.06] bg-slate-950/60 hover:border-sky-400/40 transition-colors">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-sky-400 font-semibold">{node.title}</span>
                    <span className="text-[10px] text-slate-500">{node.shape}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{node.desc}</p>
                </div>
                {i < dagNodes.length - 1 && (
                  <div className="flex justify-center py-1">
                    <div className="w-0.5 h-3 bg-gradient-to-b from-sky-400/60 to-transparent" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Right: Real-Time YOLO Entity Ledger (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
                <Database className="w-4 h-4 text-emerald-400" /> Live YOLOv8 Tensor Ledger
              </span>
              <span className="text-[10px] font-mono text-emerald-400 animate-pulse">
                FRAME #{frame?.frame_id || 1042}
              </span>
            </div>

            {/* Table of active detections */}
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400 text-[10px] uppercase">
                    <th className="pb-2">CLASS</th>
                    <th className="pb-2">CONF</th>
                    <th className="pb-2">CX / CY (NORM)</th>
                    <th className="pb-2">BBOX (W × H)</th>
                    <th className="pb-2">TRACK ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {frame?.detections?.map((d, i) => (
                    <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-2.5">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            d.class_name === 'GLOVEBOX-01'
                              ? 'bg-sky-500/10 text-sky-400 border border-sky-400/20'
                              : d.class_name === 'CENTRIFUGE-02'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-400/20'
                              : d.class_name === 'SMP-RED'
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-400/20'
                              : 'bg-amber-500/10 text-amber-400 border border-amber-400/20'
                          }`}
                        >
                          {d.class_name}
                        </span>
                      </td>
                      <td className="py-2.5 text-emerald-400 font-bold">
                        {(d.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-slate-300">
                        {d.bbox.cx.toFixed(3)}, {d.bbox.cy.toFixed(3)}
                      </td>
                      <td className="py-2.5 text-slate-400">
                        {Math.round(d.bbox.w)} × {Math.round(d.bbox.h)} px
                      </td>
                      <td className="py-2.5 text-slate-500 text-[10px]">
                        TRK-00{i + 1}
                      </td>
                    </tr>
                  ))}
                  {(!frame?.detections || frame.detections.length === 0) && (
                    <tr>
                      <td colSpan={5} className="py-4 text-center text-slate-500 text-xs">
                        No active detections in current frame buffer
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Architecture Metadata */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
              <span className="text-[10px] font-mono text-slate-500 block">MODEL PARAMETERS</span>
              <span className="text-lg font-mono font-bold text-sky-400 mt-1 block">3.2M Params</span>
              <span className="text-[10px] text-slate-400">8.7 GFLOPs @ 640×640</span>
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
              <span className="text-[10px] font-mono text-slate-500 block">INFERENCE RUNTIME</span>
              <span className="text-lg font-mono font-bold text-emerald-400 mt-1 block">ONNX Runtime</span>
              <span className="text-[10px] text-slate-400">WASM / WebGL Execution</span>
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
              <span className="text-[10px] font-mono text-slate-500 block">QUANTIZATION</span>
              <span className="text-lg font-mono font-bold text-amber-400 mt-1 block">INT8 Edge</span>
              <span className="text-[10px] text-slate-400">6.3 MB Disk Footprint</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default AIVision;
