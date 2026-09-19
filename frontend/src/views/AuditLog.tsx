import React, { useState } from 'react';
import { ScrollText, Filter, Download, AlertTriangle, Info, CheckCircle2, ShieldAlert } from 'lucide-react';
import { AUDIT_EVENTS } from '../data/mockData';
import { AuditEvent } from '../types/detections';

export const AuditLog: React.FC = () => {
  const [filter, setFilter] = useState<string>('ALL');
  const [events] = useState<AuditEvent[]>(AUDIT_EVENTS);

  const filtered = filter === 'ALL'
    ? events
    : events.filter(e => e.category === filter);

  const handleExportCSV = () => {
    const headers = ['ID', 'Timestamp', 'Category', 'Severity', 'Step_Index', 'Message'];
    const rows = events.map(e => [
      e.id,
      `"${e.timestamp}"`,
      e.category,
      e.severity || 'info',
      e.step_index,
      `"${e.message.replace(/"/g, '""')}"`,
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ASTRA_HAR_AUDIT_LOG.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ASTRA_HAR_AUDIT_LOG.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <ScrollText className="w-5 h-5 text-sky-400" />
            <h1 className="text-xl font-bold font-sans text-white">Station Audit Log & Event Ledger</h1>
          </div>
          <p className="text-xs text-slate-400">
            Immutable chronological record of astronaut interactions, YOLO classifications, and FSM sequence deviations.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <button
            onClick={handleExportCSV}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-white/10 bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
          >
            <Download className="w-3.5 h-3.5" /> CSV
          </button>
          <button
            onClick={handleExportJSON}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold transition-colors shadow-lg shadow-sky-500/20"
          >
            <Download className="w-3.5 h-3.5" /> JSON
          </button>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 text-xs font-mono text-slate-400 mr-2">
          <Filter className="w-3.5 h-3.5" /> FILTER:
        </div>
        {['ALL', 'ACTION', 'OBJECT', 'WARNING', 'SYSTEM'].map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
              filter === cat
                ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                : 'bg-slate-900/80 border border-white/10 text-slate-400 hover:text-white hover:bg-slate-800'
            }`}
          >
            {cat}
          </button>
        ))}
        <span className="ml-auto text-xs font-mono text-slate-500">
          Showing {filtered.length} of {events.length} events
        </span>
      </div>

      {/* Timeline List */}
      <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 space-y-2 max-h-[600px] overflow-y-auto">
        {filtered.map(e => (
          <div
            key={e.id}
            className={`p-3 rounded-lg border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
              e.severity === 'error'
                ? 'bg-rose-500/10 border-rose-500/30'
                : e.severity === 'warn'
                ? 'bg-amber-500/10 border-amber-500/30'
                : 'bg-slate-950/50 border-white/[0.04] hover:border-white/10'
            }`}
          >
            <div className="flex items-start sm:items-center gap-3">
              <span className="mt-0.5 sm:mt-0 font-mono text-xs text-slate-500 shrink-0">
                {e.timestamp}
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold shrink-0 ${
                  e.category === 'WARNING'
                    ? 'bg-amber-400/20 text-amber-400 border border-amber-400/30'
                    : e.category === 'ACTION'
                    ? 'bg-sky-400/20 text-sky-400 border border-sky-400/30'
                    : e.category === 'OBJECT'
                    ? 'bg-emerald-400/20 text-emerald-400 border border-emerald-400/30'
                    : 'bg-purple-400/20 text-purple-400 border border-purple-400/30'
                }`}
              >
                {e.category}
              </span>
              <p className="text-xs text-slate-200 font-sans leading-snug">
                {e.message}
              </p>
            </div>

            <div className="flex items-center gap-2 self-end sm:self-center shrink-0">
              <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-2 py-0.5 rounded border border-white/5">
                STEP {e.step_index + 1}
              </span>
              {e.severity === 'error' ? (
                <ShieldAlert className="w-4 h-4 text-rose-400" />
              ) : e.severity === 'warn' ? (
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-emerald-400/60" />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
export default AuditLog;
