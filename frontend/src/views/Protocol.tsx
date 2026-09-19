import React, { useState } from 'react';
import { Plus, Download, Trash2, CheckCircle, FileText, Sliders, ShieldCheck } from 'lucide-react';
import { PROTOCOL_STEPS } from '../data/mockData';
import { ProtocolStep } from '../types/detections';

export const Protocol: React.FC = () => {
  const [steps, setSteps] = useState<ProtocolStep[]>(PROTOCOL_STEPS);
  const [missionId, setMissionId] = useState('ISRO-PS-26174-MSG-01');
  const [protocolName, setProtocolName] = useState('Centrifuge Reagent Separation Protocol');
  const [savedNotice, setSavedNotice] = useState(false);

  const handleUpdate = (id: number, field: keyof ProtocolStep, value: any) => {
    setSteps(prev => prev.map(s => (s.id === id ? { ...s, [field]: value } : s)));
  };

  const handleAddStep = () => {
    const nextId = steps.length ? Math.max(...steps.map(s => s.id)) + 1 : 1;
    const newStep: ProtocolStep = {
      id: nextId,
      action: 'Verify containment seal',
      object: 'GLOVEBOX-01',
      status: 'pending',
      confidence: 0.85,
      guidance: 'Inspect visual pressure indicator ring before proceeding with extraction.',
    };
    setSteps([...steps, newStep]);
  };

  const handleDeleteStep = (id: number) => {
    setSteps(prev => prev.filter(s => s.id !== id));
  };

  const handleExport = () => {
    const payload = {
      mission_id: missionId,
      protocol_name: protocolName,
      version: '1.4-ORBIT',
      exported_at: new Date().toISOString(),
      classes: ['GLOVEBOX-01', 'CENTRIFUGE-02', 'SMP-RED', 'SMP-YEL'],
      steps,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ASTRA_HAR_PROTOCOL_${missionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 3000);
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Header & Mission Metadata */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">Flight Protocol Builder</h1>
          </div>
          <p className="text-xs text-slate-400">
            Pre-flight workflow authoring suite for ISS ground directors. Defines YOLO detection dependencies & FSM rules.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {savedNotice && (
            <span className="text-xs font-mono text-emerald-400 flex items-center gap-1.5 animate-fade-in">
              <CheckCircle className="w-4 h-4" /> JSON Exported!
            </span>
          )}
          <button
            onClick={handleExport}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs font-mono transition-colors shadow-lg shadow-sky-500/20"
          >
            <Download className="w-4 h-4" />
            EXPORT PROTOCOL JSON
          </button>
        </div>
      </div>

      {/* Protocol Configuration Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <label className="text-xs font-mono text-slate-400 block mb-1">PROTOCOL TITLE</label>
          <input
            type="text"
            value={protocolName}
            onChange={e => setProtocolName(e.target.value)}
            className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-400 font-sans"
          />
        </div>
        <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
          <label className="text-xs font-mono text-slate-400 block mb-1">MISSION IDENTIFIER</label>
          <input
            type="text"
            value={missionId}
            onChange={e => setMissionId(e.target.value)}
            className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-sky-400 focus:outline-none focus:border-sky-400 font-mono"
          />
        </div>
      </div>

      {/* Steps List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
            Sequential FSM Nodes ({steps.length} Steps)
          </span>
          <span className="text-[11px] font-mono text-slate-500">
            Ordered sequence strictly validated during live operations
          </span>
        </div>

        {steps.map((step, idx) => (
          <div
            key={step.id}
            className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 space-y-3 transition-all hover:border-white/20"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] pb-3">
              <div className="flex items-center gap-3">
                <span className="px-2.5 py-1 rounded-md bg-sky-500/10 border border-sky-400/20 text-sky-400 font-mono text-xs font-bold">
                  STEP 0{idx + 1}
                </span>
                <input
                  type="text"
                  value={step.action}
                  onChange={e => handleUpdate(step.id, 'action', e.target.value)}
                  className="bg-transparent font-semibold text-slate-200 text-sm focus:outline-none focus:border-b focus:border-sky-400 px-1 py-0.5"
                />
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-400">Target Object:</span>
                  <select
                    value={step.object}
                    onChange={e => handleUpdate(step.id, 'object', e.target.value)}
                    className="bg-slate-950 border border-white/10 text-xs font-mono rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-sky-400"
                  >
                    <option value="GLOVEBOX-01">GLOVEBOX-01</option>
                    <option value="CENTRIFUGE-02">CENTRIFUGE-02</option>
                    <option value="SMP-RED">SMP-RED</option>
                    <option value="SMP-YEL">SMP-YEL</option>
                  </select>
                </div>
                <button
                  onClick={() => handleDeleteStep(step.id)}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                  title="Delete Step"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
              <div className="md:col-span-8">
                <label className="text-[10px] font-mono text-slate-500 block mb-1">
                  ASTRONAUT AUDIO & DISPLAY GUIDANCE
                </label>
                <textarea
                  value={step.guidance}
                  onChange={e => handleUpdate(step.id, 'guidance', e.target.value)}
                  rows={2}
                  className="w-full bg-slate-950/80 border border-white/10 rounded-lg p-2 text-xs text-slate-300 focus:outline-none focus:border-sky-400 font-sans resize-none"
                />
              </div>

              <div className="md:col-span-4 bg-slate-950/50 p-3 rounded-lg border border-white/5 space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-sky-400" /> CONF THRESHOLD
                  </span>
                  <span className="text-sky-400 font-bold">{(step.confidence || 0.85).toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={step.confidence || 0.85}
                  onChange={e => handleUpdate(step.id, 'confidence', parseFloat(e.target.value))}
                  className="w-full accent-sky-400 h-1 bg-slate-800 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>
        ))}

        <button
          onClick={handleAddStep}
          className="w-full py-3 rounded-xl border border-dashed border-white/20 hover:border-sky-400/50 bg-slate-900/30 hover:bg-sky-500/5 text-slate-300 hover:text-sky-300 font-mono text-xs flex items-center justify-center gap-2 transition-all"
        >
          <Plus className="w-4 h-4" /> ADD VALIDATION STEP TO FSM
        </button>
      </div>
    </div>
  );
};
export default Protocol;
