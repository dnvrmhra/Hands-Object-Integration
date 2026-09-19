import { useState, useEffect, useRef, useCallback } from 'react';
import { DetectionFrame } from '../types/detections';

interface UseYOLODetectionsReturn {
  frame: DetectionFrame | null;
  isConnected: boolean;
  error: string | null;
}

export function useYOLODetections(url: string = 'ws://localhost:8000/ws/detections'): UseYOLODetectionsReturn {
  const [frame, setFrame] = useState<DetectionFrame | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(event.data as string) as DetectionFrame;
          setFrame(data);
        } catch {
          setError('Failed to parse detection frame');
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setIsConnected(false);
        // Auto-reconnect after 3 seconds
        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, 3000);
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setError('WebSocket connection error');
        setIsConnected(false);
      };
    } catch (err) {
      setError(`Failed to connect: ${String(err)}`);
    }
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { frame, isConnected, error };
}