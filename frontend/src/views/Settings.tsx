import React, { useState } from 'react';
import { Settings as SettingsIcon, VolumeX, Volume2, Cpu, HardDrive, Shield, CheckCircle2 } from 'lucide-react';

export const Settings: React.FC = () => {
  const [silentMode, setSilentMode] = useState(true);
  const [voiceAlerts, setVoiceAlerts] = useState(false);
  const [audioTones, setAudioTones] = useState(false);

  const [hardwareTarget, setHardwareTarget] = useState('NPU');
  const [modelVariant, setModelVariant] = useState('YOLOv8n (Nano - 3.2M)');
  const [confThreshold, setConfThreshold] = useState(0.60);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [downlinkMode, setDownlinkMode] = useState('EVENTS_ONLY');
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">Station Configuration & Compliance</h1>
          </div>
          <p className="text-xs text-slate-400">
            Enforce ISS Flight Rules, hardware execution target acceleration, sound policies, and telemetry parameters.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {savedSuccess && (
            <span className="text-xs font-mono text-emerald-400 flex items-center gap-1.5 animate-fade-in">
              <CheckCircle2 className="w-4 h-4" /> Parameters Synced to Station!
            </span>
          )}
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs font-mono transition-colors shadow-lg shadow-sky-500/20"
          >
            SAVE CONFIGURATION
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sound Policy & ISS Compliance */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
            <VolumeX className="w-4 h-4 text-rose-400" /> Acoustic Policy (ISS Flight Rule §4.2.1)
          </span>
          <p className="text-xs text-slate-400 leading-relaxed">
            Astronaut cabin sleep and focus rules require non-intrusive operations. 
            Auditory alarms must be muted during non-critical research cycles.
          </p>

          <div className="space-y-3 pt-2">
            <label className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-slate-950/60 cursor-pointer">
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Silent Operation Mode</span>
                <span className="text-[11px] text-slate-400">Suppresses all speaker beeps & chime alarms</span>
              </div>
              <input
                type="checkbox"
                checked={silentMode}
                onChange={e => {
                  setSilentMode(e.target.checked);
                  if (e.target.checked) {
                    setVoiceAlerts(false);
                    setAudioTones(false);
                  }
                }}
                className="w-4 h-4 accent-sky-400"
              />
            </label>

            <label
              className={`flex items-center justify-between p-3 rounded-lg border border-white/5 bg-slate-950/60 cursor-pointer ${
                silentMode ? 'opacity-40 pointer-events-none' : ''
              }`}
            >
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Synthesized Voice Guidance</span>
                <span className="text-[11px] text-slate-400">Speech cues for active protocol steps</span>
              </div>
              <input
                type="checkbox"
                checked={voiceAlerts}
                onChange={e => setVoiceAlerts(e.target.checked)}
                className="w-4 h-4 accent-sky-400"
              />
            </label>

            <label
              className={`flex items-center justify-between p-3 rounded-lg border border-white/5 bg-slate-950/60 cursor-pointer ${
                silentMode ? 'opacity-40 pointer-events-none' : ''
              }`}
            >
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Deviation Buzzer Tone</span>
                <span className="text-[11px] text-slate-400">Acoustic alert when out-of-order action occurs</span>
              </div>
              <input
                type="checkbox"
                checked={audioTones}
                onChange={e => setAudioTones(e.target.checked)}
                className="w-4 h-4 accent-sky-400"
              />
            </label>
          </div>
        </div>

        {/* Edge Hardware Acceleration */}
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 space-y-4">
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-sky-400" /> Edge NPU Hardware Target
          </span>
          <p className="text-xs text-slate-400 leading-relaxed">
            Select on-device acceleration runtime. Browser fallback uses WebGL/WASM when dedicated NPU is busy.
          </p>

          <div className="grid grid-cols-2 gap-2 pt-2">
            {[
              { id: 'NPU', label: 'Dedicated Edge NPU (14ms)' },
              { id: 'CUDA', label: 'NVIDIA Jetson CUDA (16ms)' },
              { id: 'ONNX_WEB', label: 'Browser WebGL (28ms)' },
              { id: 'CPU', label: 'x86_64 Fallback (62ms)' },
            ].map(hw => (
              <button
                key={hw.id}
                onClick={() => setHardwareTarget(hw.id)}
                className={`p-3 rounded-lg border text-left font-mono text-xs transition-colors ${
                  hardwareTarget === hw.id
                    ? 'bg-sky-500/10 border-sky-400 text-sky-400 font-bold'
                    : 'bg-slate-950/60 border-white/5 text-slate-400 hover:border-white/20'
                }`}
              >
                {hw.label}
              </button>
            ))}
          </div>

          <div className="pt-2 space-y-3">
            <div>
              <label className="text-[10px] font-mono text-slate-400 block mb-1">MODEL TOPOLOGY</label>
              <select
                value={modelVariant}
                onChange={e => setModelVariant(e.target.value)}
                className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-sky-400"
              >
                <option value="YOLOv8n (Nano - 3.2M)">YOLOv8n (Nano - 3.2M params • Optimized)</option>
                <option value="YOLOv8s (Small - 11.2M)">YOLOv8s (Small - 11.2M params • High mAP)</option>
                <option value="YOLOv8m (Medium - 25.9M)">YOLOv8m (Medium - 25.9M params • Benchmark)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">CONF CUTOFF</span>
                  <span className="text-sky-400 font-bold">{confThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={confThreshold}
                  onChange={e => setConfThreshold(parseFloat(e.target.value))}
                  className="w-full accent-sky-400 h-1.5 bg-slate-800 rounded cursor-pointer"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">NMS IoU CUTOFF</span>
                  <span className="text-sky-400 font-bold">{iouThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.05"
                  value={iouThreshold}
                  onChange={e => setIouThreshold(parseFloat(e.target.value))}
                  className="w-full accent-sky-400 h-1.5 bg-slate-800 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Settings;
