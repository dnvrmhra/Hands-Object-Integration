import { useState, useEffect, useRef } from 'react';
import { DetectionFrame } from '../types/detections';
import { MOCK_DETECTION_SEQUENCES } from '../data/mockData';

interface UseMockDetectionsReturn {
  frame: DetectionFrame | null;
  isRunning: boolean;
  frameIndex: number;
}

export function useMockDetections(activeStep: number): UseMockDetectionsReturn {
  const [frame, setFrame] = useState<DetectionFrame | null>(null);
  const [isRunning, setIsRunning] = useState(true);
  const [frameIndex, setFrameIndex] = useState(0);
  const stepRef = useRef(activeStep);
  const indexRef = useRef(0);

  useEffect(() => {
    stepRef.current = activeStep;
    indexRef.current = 0;
    setFrameIndex(0);
    setIsRunning(true);
  }, [activeStep]);

  useEffect(() => {
    const sequence = MOCK_DETECTION_SEQUENCES[Math.min(activeStep, MOCK_DETECTION_SEQUENCES.length - 1)];
    if (!sequence || sequence.length === 0) return;

    const interval = setInterval(() => {
      const idx = indexRef.current % sequence.length;
      setFrame(sequence[idx]);
      setFrameIndex(idx);
      indexRef.current += 1;
    }, 100); // 10 fps

    return () => clearInterval(interval);
  }, [activeStep]);

  return { frame, isRunning, frameIndex };
}