import React, { useRef, useEffect, useState } from 'react';
import { DetectionFrame } from '../types/detections';
import { Camera, RefreshCw, Sparkles, MoveRight, CheckCircle2, AlertCircle } from 'lucide-react';

export type CameraSourceMode = 'LIVE_WEBCAM' | 'SIMULATION' | 'VIDEO_UPLOAD';

interface CameraViewProps {
  frame: DetectionFrame | null;
  activeStep: number;
  activeChannel: string;
  onChannelChange: (ch: string) => void;
  onFrameChange?: (frame: DetectionFrame) => void;
}

const CHANNELS = ['RGB', 'THERMAL', 'DEPTH', 'CANNY', '★ ATTN'] as const;

export const CameraView: React.FC<CameraViewProps> = ({
  frame,
  activeStep,
  activeChannel,
  onChannelChange,
  onFrameChange,
}) => {
  const [mode, setMode] = useState<CameraSourceMode>('LIVE_WEBCAM');
  const [streamError, setStreamError] = useState<boolean>(false);
  const [retryKey, setRetryKey] = useState<number>(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const movement = frame?.movement;
  const isMoving = movement?.state === 'TRANSPORTING';
  const isGrasped = movement?.state === 'GRASPED';
  const isPlaced = movement?.is_placed_recently;

  return (
    <div className="flex flex-col gap-2.5 w-full">
      {/* Top Source Mode Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-lg bg-slate-950/80 border border-white/10">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setMode('LIVE_WEBCAM')}
            className={`px-3 py-1.5 rounded text-xs font-mono font-medium flex items-center gap-1.5 transition-colors ${
              mode === 'LIVE_WEBCAM'
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Live Vision (Cans, Bottles, Phones & Humans)
          </button>
          <button
            onClick={() => setMode('SIMULATION')}
            className={`px-3 py-1.5 rounded text-xs font-mono font-medium transition-colors ${
              mode === 'SIMULATION' ? 'bg-slate-700 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Synthetic Simulation
          </button>
        </div>

        {mode === 'LIVE_WEBCAM' && (
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            {/* Detection Counts */}
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-900 border border-white/10 text-[11px]">
              <span className="text-slate-400">Detections:</span>
              <span className="text-cyan-400 font-bold">Persons: {frame?.counts?.humans ?? 0}</span>
              <span className="text-slate-500">•</span>
              <span className="text-emerald-400 font-bold">Bottles: {frame?.counts?.bottles ?? 0}</span>
              <span className="text-slate-500">•</span>
              <span className="text-amber-400 font-bold">Cans: {frame?.counts?.cans ?? 0}</span>
              <span className="text-slate-500">•</span>
              <span className="text-sky-400 font-bold">Phones: {frame?.counts?.phones ?? 0}</span>
            </div>

            {/* Human Activity State Badge */}
            {frame?.counts?.human_state && frame.counts.human_state !== 'NONE' && (
              <div
                className={`px-2 py-0.5 rounded text-[11px] font-bold border flex items-center gap-1 ${
                  frame.counts.human_state === 'WALKING'
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 animate-pulse'
                    : frame.counts.human_state === 'MOVING'
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                    : 'bg-slate-800 text-slate-300 border-white/10'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-current" />
                HUMAN: {frame.counts.human_state}
              </div>
            )}

            {/* Lighting Status Badge */}
            {frame?.lighting && frame.lighting !== 'NORMAL' && (
              <div className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-400/40 text-[10px]">
                {frame.lighting}
              </div>
            )}

            <button
              onClick={() => {
                setStreamError(false);
                setRetryKey(k => k + 1);
              }}
              className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-white/10 hover:border-cyan-400/40 hover:text-white flex items-center gap-1 text-[11px]"
            >
              <RefreshCw className="w-3 h-3" />
              Reconnect
            </button>
          </div>
        )}
      </div>

      {/* Main Viewport */}
      <div className="relative w-full rounded-lg overflow-hidden border border-white/[0.08] bg-slate-950" style={{ height: '360px' }}>
        {mode === 'LIVE_WEBCAM' ? (
          <div className="relative w-full h-full bg-slate-950 flex items-center justify-center">
            {!streamError ? (
              <img
                key={retryKey}
                src="http://localhost:8000/video_feed"
                alt="ASTRA Live Feed"
                className="w-full h-full object-contain"
                onError={() => setStreamError(true)}
                onLoad={() => setStreamError(false)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center p-6 text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                  <AlertCircle className="w-6 h-6 animate-pulse" />
                </div>
                <div className="text-sm font-mono font-bold text-slate-200">
                  VISION SERVER CONNECTING...
                </div>
                <p className="text-xs text-slate-400 max-w-sm">
                  Live vision server streaming webcam with person and custom object detection:
                </p>
                <code className="text-[11px] font-mono bg-slate-900 px-3 py-1.5 rounded border border-white/10 text-cyan-400">
                  python -m uvicorn inference.inference_server:app --port 8000
                </code>
                <button
                  onClick={() => {
                    setStreamError(false);
                    setRetryKey(k => k + 1);
                  }}
                  className="px-3.5 py-1.5 rounded text-xs font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-400/40 hover:bg-cyan-500/30 transition-colors"
                >
                  Retry Stream Ingest
                </button>
              </div>
            )}

            {/* Top-Left Live Badge */}
            <div className="absolute top-2 left-2 px-2.5 py-1 rounded bg-slate-950/85 border border-cyan-400/40 text-[10px] font-mono text-cyan-400 flex items-center gap-1.5 shadow-lg backdrop-blur-md">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              LIVE • CANS, BOTTLES, PHONES & HUMANS
            </div>

            {/* Top-Right Active State Badge */}
            {movement && (
              <div
                className={`absolute top-2 right-2 px-2.5 py-1.5 rounded text-[11px] font-mono flex items-center gap-1.5 shadow-lg backdrop-blur-md border ${
                  isMoving
                    ? 'bg-amber-500/20 border-amber-400/60 text-amber-300 animate-pulse font-bold'
                    : isPlaced
                    ? 'bg-emerald-500/20 border-emerald-400/60 text-emerald-300 font-bold'
                    : isGrasped
                    ? 'bg-cyan-500/20 border-cyan-400/60 text-cyan-300 font-bold'
                    : 'bg-slate-900/85 border-white/10 text-slate-400'
                }`}
              >
                {isMoving ? (
                  <>
                    <MoveRight className="w-3.5 h-3.5 text-amber-400" />
                    <span>MOVING {movement.active_object}: {movement.distance_cm} cm</span>
                  </>
                ) : isPlaced ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span>{movement.active_object} PLACED AT NEW POSITION</span>
                  </>
                ) : isGrasped ? (
                  <>
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    <span>GRASPED {movement.active_object}</span>
                  </>
                ) : (
                  <span>MONITORING POSITIONS</span>
                )}
              </div>
            )}

            {/* Bottom-Left Real-Time Movement Readout */}
            {isMoving && movement && (
              <div className="absolute bottom-2 left-2 px-3 py-1.5 rounded bg-slate-950/90 border border-amber-400/40 text-[10px] font-mono text-slate-200 flex items-center gap-3 shadow-lg backdrop-blur-md">
                <div>
                  <span className="text-slate-400">DISPLACEMENT: </span>
                  <span className="text-amber-400 font-bold font-mono text-xs">{movement.distance_cm} cm</span>
                </div>
                <div className="w-px h-3 bg-white/20" />
                <div>
                  <span className="text-slate-400">SPEED: </span>
                  <span className="text-cyan-400 font-bold font-mono text-xs">{movement.speed_cm_s} cm/s</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center p-6 text-center space-y-3">
            <div className="text-xs font-mono font-bold text-slate-300">
              SYNTHETIC TEST SIMULATION
            </div>
            <p className="text-[11px] text-slate-400 max-w-sm">
              Procedural simulation of microgravity container manipulation protocol.
            </p>
            <button
              onClick={() => setMode('LIVE_WEBCAM')}
              className="px-3 py-1.5 rounded text-xs font-mono bg-cyan-500 text-slate-950 font-bold"
            >
              Switch back to Live Webcam
            </button>
          </div>
        )}
      </div>

      {/* Channel Switcher */}
      <div className="flex gap-1.5">
        {CHANNELS.map(ch => (
          <button
            key={ch}
            onClick={() => onChannelChange(ch)}
            className={`flex-1 py-1.5 rounded text-xs font-mono font-medium transition-all duration-200 ${
              activeChannel === ch
                ? 'bg-cyan-400/20 border border-cyan-400/50 text-cyan-400'
                : 'bg-slate-800/60 border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/20'
            }`}
          >
            {ch}
          </button>
        ))}
      </div>
    </div>
  );
};