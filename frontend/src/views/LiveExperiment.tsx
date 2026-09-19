import React, { useState } from 'react';
import { CameraView } from '../components/CameraView';
import { Glovebox3DTwin } from '../components/Glovebox3DTwin';
import { SequenceTimeline } from '../components/SequenceTimeline';
import { ValidationControls } from '../components/ValidationControls';
import { OrientationPanel } from '../components/OrientationPanel';
import { BandwidthPanel } from '../components/BandwidthPanel';
import { useLiveInference } from '../hooks/useLiveInference';
import { useMockDetections } from '../hooks/useMockDetections';
import { Activity, ShieldAlert, Cpu, Sparkles, CheckCircle2, MoveRight, Layers } from 'lucide-react';

interface LiveExperimentProps {
  activeStep: number;
  setActiveStep: (step: number) => void;
  hasDeviation: boolean;
  setHasDeviation: (dev: boolean) => void;
}

export const LiveExperiment: React.FC<LiveExperimentProps> = ({
  activeStep,
  setActiveStep,
  hasDeviation,
  setHasDeviation,
}) => {
  const [activeChannel, setActiveChannel] = useState<string>('RGB');

  // Connect to live vision & movement telemetry stream from backend
  const liveFeed = useLiveInference(true);
  const mockFeed = useMockDetections(activeStep);

  const frame = liveFeed.frame || mockFeed.frame;
  const movement = frame?.movement;
  const attitude = frame?.attitude_3d || (frame?.hands?.[0]?.attitude_3d);

  const handleValidate = () => {
    if (activeStep < 6) {
      setActiveStep(activeStep + 1);
    }
    setHasDeviation(false);
  };

  const handleDeviation = () => {
    setHasDeviation(true);
  };

  const handleSkip = () => {
    if (activeStep < 6) {
      setActiveStep(activeStep + 1);
    }
  };

  const currentConfidence = frame?.detections?.length
    ? Math.max(...frame.detections.map(d => d.confidence))
    : 0.95;

  const isMoving = movement?.state === 'TRANSPORTING';
  const isPlaced = movement?.is_placed_recently;
  const isGrasped = movement?.state === 'GRASPED';

  return (
    <div className="space-y-4 pb-8">
      {/* Top Banner Status */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-slate-200 flex items-center gap-2">
              LIVE VISION DECK • BOTTLES, CANS, PHONES & HUMANS
              <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-400/30 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                LIVE WEBCAM ACTIVE
              </span>
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {isMoving ? (
                <span className="text-amber-300 font-bold flex items-center gap-1.5">
                  <MoveRight className="w-3.5 h-3.5 animate-pulse" />
                  MOVING {movement?.active_object}: {movement?.distance_cm} cm ({movement?.speed_cm_s} cm/s)
                </span>
              ) : isPlaced ? (
                <span className="text-emerald-300 font-bold flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  {movement?.active_object} SUCCESSFULLY RELOCATED (+{movement?.distance_cm} cm)
                </span>
              ) : isGrasped ? (
                <span className="text-cyan-300 font-bold">
                  GRASPED {movement?.active_object} • READY FOR TRANSPORT
                </span>
              ) : (
                `Bottles: ${frame?.counts?.bottles ?? 0} | Cans: ${frame?.counts?.cans ?? 0} | Phones: ${frame?.counts?.phones ?? 0} • Human: ${frame?.counts?.human_state ?? 'IDLE'}`
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-white/5">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">FPS:</span>
            <span className="text-emerald-400 font-bold">{frame?.fps || 30}</span>
          </div>
          {isMoving ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/20 border border-amber-400/60 text-amber-300 font-bold animate-pulse">
              <MoveRight className="w-3.5 h-3.5" />
              RELOCATION IN PROGRESS
            </div>
          ) : isPlaced ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/20 border border-emerald-400/60 text-emerald-300 font-bold">
              <CheckCircle2 className="w-3.5 h-3.5" />
              ITEM PLACED
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              NOMINAL MONITORING
            </div>
          )}
        </div>
      </div>

      {/* Main Grid: Left Camera & Overlays (60%), Right 3D Twin & Timeline (40%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column (7 cols on lg) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
            <div className="flex items-center justify-between pb-3 mb-2 border-b border-white/[0.06]">
              <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
                Live Optical Feed & Hand Trajectory Overlay
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-400/20">
                CAN & BOTTLE VISION
              </span>
            </div>

            <CameraView
              frame={frame}
              activeStep={activeStep}
              activeChannel={activeChannel}
              onChannelChange={setActiveChannel}
            />
          </div>

          {/* Can & Bottle Movement Ledger Card */}
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4 space-y-2.5">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
              <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                Recent Can & Bottle Relocation Ledger
              </span>
              <span className="text-[10px] font-mono text-slate-500">
                {movement?.history?.length || 0} COMPLETED TRANSFERS
              </span>
            </div>

            {movement?.history && movement.history.length > 0 ? (
              <div className="space-y-1.5 max-h-36 overflow-y-auto font-mono text-xs">
                {movement.history.map((evt) => (
                  <div
                    key={evt.id}
                    className="flex items-center justify-between p-2 rounded bg-slate-950/70 border border-white/[0.04] hover:border-cyan-400/30 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span className="font-bold text-slate-200">{evt.object_name}</span>
                      <span className="text-slate-500 text-[10px]">{evt.timestamp}</span>
                    </div>
                    <div className="flex items-center gap-3 text-slate-400 text-[11px]">
                      <span>
                        Distance: <strong className="text-cyan-400">{evt.distance_cm} cm</strong>
                      </span>
                      <span>
                        Time: <strong className="text-amber-400">{evt.duration_s}s</strong>
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] border border-emerald-400/20">
                        PLACED
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center text-xs font-mono text-slate-500">
                No items moved yet. Bring a can or bottle into webcam view and move it to track displacement.
              </div>
            )}
          </div>

          {/* Sub-strip: Microgravity Attitude & Bandwidth Reduction */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
              <OrientationPanel attitude={attitude} />
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
              <BandwidthPanel />
            </div>
          </div>
        </div>

        {/* Right Column (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-4">
          {/* 3D Kinematic Digital Twin */}
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
                3D Kinematic Digital Twin
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-400/20">
                6-DOF ORBIT
              </span>
            </div>
            <div className="h-64 rounded-lg overflow-hidden border border-white/[0.06] bg-slate-950">
              <Glovebox3DTwin activeStep={activeStep} attitude={attitude} />
            </div>
          </div>

          {/* Sequence Timeline */}
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
                7-Step Protocol Timeline
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                STEP {activeStep + 1} OF 7
              </span>
            </div>
            <SequenceTimeline activeStep={activeStep} onStepClick={setActiveStep} />
          </div>

          {/* Validation & Deviation Controls */}
          <div className="rounded-xl border border-white/[0.08] bg-slate-900/60 backdrop-blur-md p-4">
            <div className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider mb-3">
              Operator Verification & FSM Controls
            </div>
            <ValidationControls
              activeStep={activeStep}
              onValidate={handleValidate}
              onDeviation={handleDeviation}
              onSkip={handleSkip}
              confidence={currentConfidence}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
export default LiveExperiment;
