import { useState, useEffect, useRef } from 'react';
import { DetectionFrame, HOISummary, HandData } from '../types/detections';

export interface LiveStreamState {
  frame: DetectionFrame | null;
  isConnected: boolean;
  isStreaming: boolean;
  cameraSource: string;
  hoi: HOISummary;
  hands: HandData[];
  error: string | null;
}

const DEFAULT_HOI: HOISummary = {
  state: 'IDLE',
  target_object: null,
  confidence: 0.95,
  proximity_cm: 0.0,
  action_label: 'STANDBY',
};

export function useLiveInference(enabled: boolean = false, wsUrl: string = 'ws://localhost:8000/ws/live') {
  const [state, setState] = useState<LiveStreamState>({
    frame: null,
    isConnected: false,
    isStreaming: false,
    cameraSource: 'DISCONNECTED',
    hoi: DEFAULT_HOI,
    hands: [],
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setState(prev => ({ ...prev, isConnected: false, isStreaming: false }));
      return;
    }

    let isMounted = true;

    function connect() {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted) return;
          setState(prev => ({
            ...prev,
            isConnected: true,
            isStreaming: true,
            error: null,
          }));
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const data = JSON.parse(event.data);
            setState(prev => ({
              ...prev,
              frame: {
                frame_id: data.frame_id || 0,
                timestamp: data.timestamp || Date.now() / 1000,
                fps: data.fps || 25,
                detections: data.detections || [],
                bandwidth_saved_pct: 98.2,
                active_step: data.hoi?.state === 'TRANSPORTING' ? 3 : 1,
                hoi: data.hoi || DEFAULT_HOI,
                hands: data.hands || [],
                camera_source: data.camera_source || 'LIVE_INGEST',
              },
              cameraSource: data.camera_source || 'LIVE_INGEST',
              hoi: data.hoi || DEFAULT_HOI,
              hands: data.hands || [],
            }));
          } catch (e) {
            // Ignore parse error
          }
        };

        ws.onclose = () => {
          if (!isMounted) return;
          setState(prev => ({ ...prev, isConnected: false, isStreaming: false }));
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (isMounted && enabled) {
              connect();
            }
          }, 3000);
        };

        ws.onerror = () => {
          if (!isMounted) return;
          setState(prev => ({ ...prev, error: 'Cannot connect to Python vision server (ws://localhost:8000)' }));
        };
      } catch (err: any) {
        setState(prev => ({ ...prev, error: err?.message || 'Connection failed' }));
      }
    }

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enabled, wsUrl]);

  return state;
}
