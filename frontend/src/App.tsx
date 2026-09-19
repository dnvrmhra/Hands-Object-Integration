import React, { useState } from 'react';
import { CyberTopBar } from './components/CyberTopBar';
import {
  Activity,
  Video,
  ClipboardList,
  Cpu,
  ScrollText,
  Camera,
  Gauge,
  Settings as SettingsIcon,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Views
import { Overview } from './views/Overview';
import { LiveExperiment } from './views/LiveExperiment';
import { Protocol } from './views/Protocol';
import { AIVision } from './views/AIVision';
import { AuditLog } from './views/AuditLog';
import { Optics } from './views/Optics';
import { Diagnostics } from './views/Diagnostics';
import { Settings } from './views/Settings';

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<string>('overview');
  const [activeStep, setActiveStep] = useState<number>(2); // Start on step 3 (load red)
  const [hasDeviation, setHasDeviation] = useState<boolean>(false);

  const navItems = [
    { id: 'overview', label: 'Mission Overview', icon: Activity },
    { id: 'live', label: 'Live Operational Deck', icon: Video },
    { id: 'protocol', label: 'Flight Protocol FSM', icon: ClipboardList },
    { id: 'vision', label: 'Neural Tree DAG', icon: Cpu },
    { id: 'audit', label: 'Event Audit Ledger', icon: ScrollText },
    { id: 'optics', label: 'Optics & Stream', icon: Camera },
    { id: 'diagnostics', label: 'Edge Diagnostics', icon: Gauge },
    { id: 'settings', label: 'Station Settings', icon: SettingsIcon },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-sky-200">
      {/* Cockpit Top Bar */}
      <CyberTopBar hasDeviation={hasDeviation} />

      {/* Main Layout (below 56px top header) */}
      <div className="pt-14 flex flex-1 overflow-hidden">
        {/* Sidebar Nav */}
        <aside className="w-16 sm:w-60 shrink-0 bg-slate-950/80 backdrop-blur-xl border-r border-white/[0.06] flex flex-col justify-between p-2 sm:p-3">
          <div className="space-y-1.5">
            <div className="hidden sm:block px-3 py-2 text-[10px] font-mono uppercase tracking-wider text-slate-500">
              Payload Views
            </div>
            {navItems.map(item => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveView(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
                    isActive
                      ? 'bg-sky-500 text-slate-950 font-semibold shadow-lg shadow-sky-500/20'
                      : 'text-slate-400 hover:text-white hover:bg-slate-900/60'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  <span className="hidden sm:inline text-xs font-mono font-medium truncate">
                    {item.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Sidebar Footer */}
          <div className="hidden sm:block p-3 rounded-xl border border-white/[0.04] bg-slate-900/40 text-[11px] font-mono text-slate-400 space-y-1">
            <div className="text-slate-500 text-[10px]">TEAM: UnoFlyp</div>
            <div className="text-sky-400 font-bold">ISRO PS-26174</div>
            <div className="text-[10px] text-slate-500">Columbus Workstation</div>
          </div>
        </aside>

        {/* View Content Area */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/40 via-slate-950 to-slate-950">
          <div className="max-w-7xl mx-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeView}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
                {activeView === 'overview' && (
                  <Overview onNavigate={setActiveView} activeStep={activeStep} />
                )}
                {activeView === 'live' && (
                  <LiveExperiment
                    activeStep={activeStep}
                    setActiveStep={setActiveStep}
                    hasDeviation={hasDeviation}
                    setHasDeviation={setHasDeviation}
                  />
                )}
                {activeView === 'protocol' && <Protocol />}
                {activeView === 'vision' && <AIVision activeStep={activeStep} />}
                {activeView === 'audit' && <AuditLog />}
                {activeView === 'optics' && <Optics />}
                {activeView === 'diagnostics' && <Diagnostics />}
                {activeView === 'settings' && <Settings />}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};
export default App;
