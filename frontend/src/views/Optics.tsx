import React, { useState } from 'react';
import { Camera, Sliders, HardDrive, Cpu, Film, Disc } from 'lucide-react';

export const Optics: React.FC = () => {
  const [source, setSource] = useState('ISS-COLUMBUS-CAM01');
  const [res, setRes] = useState('1080p');
  const [fps, setFps] = useState(30);
  const [codec, setCodec] = useState('H.265');
  const [bufferUsage] = useState(38.4); // percent

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">Optics & Payload Camera Ingest</h1>
          </div>
          <p className="text-xs text-slate-400">
            Configure raw sensor feeds, H.265 hardware compression, local circular ring buffers, and frame capture rates.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          OPTICAL INGEST NOMINAL
        </div>
      </div>

      {/* Control Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Source Selector */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
            <Film className="w-4 h-4 text-sky-400" /> Video Ingest Channel
          </span>

          <div className="space-y-2">
            {[
              { id: 'ISS-COLUMBUS-CAM01', name: 'Columbus Glovebox MSG-01 (Primary)' },
              { id: 'ISS-COLUMBUS-CAM02', name: 'Columbus Overhead Macro (Secondary)' },
              { id: 'SYNTHETIC-SIM-STREAM', name: 'Synthetic Procedural Generator (Local)' },
            ].map(cam => (
              <label
                key={cam.id}
                className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                  source === cam.id
                    ? 'border-sky-400 bg-sky-500/10 text-white'
                    : 'border-white/5 bg-slate-950/40 text-slate-400 hover:border-white/20'
                }`}
              >
                <input
                  type="radio"
                  name="source"
                  checked={source === cam.id}
                  onChange={() => setSource(cam.id)}
                  className="accent-sky-400"
                />
                <span className="text-xs font-sans">{cam.name}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Compression & Resolution */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-4 h-4 text-emerald-400" /> Resolution & Codec
          </span>

          <div className="space-y-4">
            <div>
              <label className="text-[10px] font-mono text-slate-400 block mb-1">OPTICAL RESOLUTION</label>
              <div className="grid grid-cols-3 gap-2">
                {['1080p', '720p', '480p'].map(r => (
                  <button
                    key={r}
                    onClick={() => setRes(r)}
                    className={`py-1.5 rounded text-xs font-mono transition-colors ${
                      res === r
                        ? 'bg-emerald-500 text-slate-950 font-bold'
                        : 'bg-slate-950 border border-white/10 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-400">FRAME RATE</span>
                <span className="text-emerald-400 font-bold">{fps} FPS</span>
              </div>
              <input
                type="range"
                min="10"
                max="60"
                step="5"
                value={fps}
                onChange={e => setFps(parseInt(e.target.value))}
                className="w-full accent-emerald-400 h-1.5 bg-slate-800 rounded cursor-pointer"
              />
            </div>

            <div>
              <label className="text-[10px] font-mono text-slate-400 block mb-1">HARDWARE CODEC</label>
              <div className="grid grid-cols-2 gap-2">
                {['H.265 (HEVC)', 'H.264 (AVC)'].map(c => (
                  <button
                    key={c}
                    onClick={() => setCodec(c.split(' ')[0])}
                    className={`py-1.5 rounded text-xs font-mono transition-colors ${
                      codec === c.split(' ')[0]
                        ? 'bg-emerald-500 text-slate-950 font-bold'
                        : 'bg-slate-950 border border-white/10 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Storage Buffer & Metrics */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-purple-400" /> Ring Buffer Telemetry
          </span>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-400">NVMe RING BUFFER</span>
                <span className="text-purple-400 font-bold">{bufferUsage}% (49.1 / 128 GB)</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div className="bg-purple-400 h-full rounded-full" style={{ width: `${bufferUsage}%` }} />
              </div>
              <span className="text-[10px] text-slate-500 mt-1 block font-mono">
                Overwrites oldest frames every 48 hours in microgravity FIFO mode
              </span>
            </div>

            <div className="p-3 rounded-lg border border-white/5 bg-slate-950/60 space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Raw Data Rate:</span>
                <span className="text-rose-400">2.4 GB / hr</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">H.265 Staged:</span>
                <span className="text-amber-400">180 MB / hr</span>
              </div>
              <div className="flex justify-between border-t border-white/5 pt-1.5 font-bold">
                <span className="text-emerald-400">ASTRA-HAR Downlink:</span>
                <span className="text-emerald-400">18.2 MB / hr</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Optics;
