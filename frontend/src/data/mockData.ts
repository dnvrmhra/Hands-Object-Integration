import { DetectionFrame, ProtocolStep, AuditEvent, SystemMetric } from '../types/detections';

// ──────────────────────────────────────────
// PROTOCOL STEPS
// ──────────────────────────────────────────
export const PROTOCOL_STEPS: ProtocolStep[] = [
  {
    id: 1,
    action: 'Initialize glovebox',
    object: 'GLOVEBOX-01',
    status: 'completed',
    confidence: 0.97,
    timestamp: 'T+00:00:12',
    guidance: 'Verify glove port seals and confirm nitrogen purge cycle is complete. Check internal pressure gauge reads 1.01 atm.',
  },
  {
    id: 2,
    action: 'Insert centrifuge rotor',
    object: 'CENTRIFUGE-02',
    status: 'completed',
    confidence: 0.93,
    timestamp: 'T+00:01:45',
    guidance: 'Align rotor tabs with centrifuge body slots. Press firmly until click is audible. Verify lock indicator is green.',
  },
  {
    id: 3,
    action: 'Load SMP-RED into centrifuge',
    object: 'SMP-RED',
    status: 'active',
    confidence: 0.89,
    guidance: 'Grasp SMP-RED sample tube by body (not cap). Insert into rotor slot A1 at 45° angle, then rotate to vertical. Confirm snap-lock.',
  },
  {
    id: 4,
    action: 'Load SMP-YEL into centrifuge',
    object: 'SMP-YEL',
    status: 'pending',
    confidence: 0.0,
    guidance: 'Mirror procedure for slot A2. Ensure SMP-YEL mass (14.3g) is balanced against SMP-RED (14.1g) — add counterweight if delta > 0.5g.',
  },
  {
    id: 5,
    action: 'Run centrifuge cycle',
    object: 'CENTRIFUGE-02',
    status: 'pending',
    confidence: 0.0,
    guidance: 'Set centrifuge to 3000 RPM, duration 12 minutes. Monitor vibration sensor — abort if amplitude exceeds 0.8g. Record cycle ID in log.',
  },
  {
    id: 6,
    action: 'Extract SMP-RED sample',
    object: 'SMP-RED',
    status: 'pending',
    confidence: 0.0,
    guidance: 'Allow centrifuge to reach full stop before opening (minimum 45s after cycle end). Extract SMP-RED using reverse of load procedure.',
  },
  {
    id: 7,
    action: 'Document results',
    object: 'GLOVEBOX-01',
    status: 'pending',
    confidence: 0.0,
    guidance: 'Photograph both sample tubes with the scale bar visible. Enter mass, color, and visual observations into the experiment log before closing glovebox.',
  },
];

// ──────────────────────────────────────────
// HELPER: build a detection frame
// ──────────────────────────────────────────
function makeFrame(
  frameId: number,
  timestamp: number,
  step: number,
  detections: Array<{ classId: number; className: DetectionFrame['detections'][0]['class_name']; conf: number; x1: number; y1: number; x2: number; y2: number }>
): DetectionFrame {
  return {
    frame_id: frameId,
    timestamp,
    fps: 10,
    bandwidth_saved_pct: 99.2,
    active_step: step,
    detections: detections.map(d => ({
      class_id: d.classId,
      class_name: d.className,
      confidence: d.conf,
      bbox: {
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
        cx: (d.x1 + d.x2) / 2,
        cy: (d.y1 + d.y2) / 2,
        w: d.x2 - d.x1,
        h: d.y2 - d.y1,
      },
    })),
  };
}

function jitter(v: number, range = 3): number {
  return v + (Math.random() - 0.5) * range;
}

function buildSequence(step: number, count = 30): DetectionFrame[] {
  const frames: DetectionFrame[] = [];
  for (let i = 0; i < count; i++) {
    const t = Date.now() + i * 100;
    switch (step) {
      case 0: // Initialize glovebox — only glovebox visible
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.94 + Math.random() * 0.03, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
        ]));
        break;
      case 1: // Insert centrifuge rotor
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.93 + Math.random() * 0.03, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 1, className: 'CENTRIFUGE-02', conf: 0.88 + Math.random() * 0.05, x1: jitter(200), y1: jitter(180), x2: jitter(380), y2: jitter(320) },
        ]));
        break;
      case 2: // Load SMP-RED
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.95 + Math.random() * 0.02, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 1, className: 'CENTRIFUGE-02', conf: 0.91 + Math.random() * 0.04, x1: jitter(200), y1: jitter(180), x2: jitter(380), y2: jitter(320) },
          { classId: 2, className: 'SMP-RED', conf: 0.87 + Math.random() * 0.06, x1: jitter(300 + i * 2), y1: jitter(250), x2: jitter(340 + i * 2), y2: jitter(310) },
        ]));
        break;
      case 3: // Load SMP-YEL
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.96 + Math.random() * 0.02, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 1, className: 'CENTRIFUGE-02', conf: 0.92 + Math.random() * 0.04, x1: jitter(200), y1: jitter(180), x2: jitter(380), y2: jitter(320) },
          { classId: 2, className: 'SMP-RED', conf: 0.93 + Math.random() * 0.03, x1: jitter(250), y1: jitter(220), x2: jitter(280), y2: jitter(300) },
          { classId: 3, className: 'SMP-YEL', conf: 0.88 + Math.random() * 0.05, x1: jitter(320 + i * 1.5), y1: jitter(250), x2: jitter(355 + i * 1.5), y2: jitter(310) },
        ]));
        break;
      case 4: // Run centrifuge cycle
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.97 + Math.random() * 0.02, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 1, className: 'CENTRIFUGE-02', conf: 0.95 + Math.random() * 0.03, x1: jitter(200), y1: jitter(180), x2: jitter(390), y2: jitter(330) },
          { classId: 2, className: 'SMP-RED', conf: 0.91 + Math.random() * 0.04, x1: jitter(248), y1: jitter(218), x2: jitter(278), y2: jitter(298) },
          { classId: 3, className: 'SMP-YEL', conf: 0.90 + Math.random() * 0.04, x1: jitter(318), y1: jitter(218), x2: jitter(348), y2: jitter(298) },
        ]));
        break;
      case 5: // Extract SMP-RED
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.96 + Math.random() * 0.02, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 1, className: 'CENTRIFUGE-02', conf: 0.92 + Math.random() * 0.04, x1: jitter(200), y1: jitter(180), x2: jitter(380), y2: jitter(320) },
          { classId: 2, className: 'SMP-RED', conf: 0.87 + Math.random() * 0.06, x1: jitter(260 + (29 - i) * 2), y1: jitter(200 + (29 - i)), x2: jitter(295 + (29 - i) * 2), y2: jitter(270 + (29 - i)) },
          { classId: 3, className: 'SMP-YEL', conf: 0.91 + Math.random() * 0.04, x1: jitter(318), y1: jitter(220), x2: jitter(348), y2: jitter(300) },
        ]));
        break;
      case 6: // Document results
        frames.push(makeFrame(i, t, step, [
          { classId: 0, className: 'GLOVEBOX-01', conf: 0.95 + Math.random() * 0.03, x1: jitter(80), y1: jitter(60), x2: jitter(560), y2: jitter(420) },
          { classId: 2, className: 'SMP-RED', conf: 0.93 + Math.random() * 0.04, x1: jitter(200), y1: jitter(280), x2: jitter(230), y2: jitter(350) },
          { classId: 3, className: 'SMP-YEL', conf: 0.92 + Math.random() * 0.04, x1: jitter(260), y1: jitter(280), x2: jitter(290), y2: jitter(350) },
        ]));
        break;
      default:
        frames.push(makeFrame(i, t, step, []));
    }
  }
  return frames;
}

export const MOCK_DETECTION_SEQUENCES: DetectionFrame[][] = Array.from({ length: 7 }, (_, i) => buildSequence(i));

// ──────────────────────────────────────────
// AUDIT EVENTS
// ──────────────────────────────────────────
export const AUDIT_EVENTS: AuditEvent[] = [
  { id: 1,  timestamp: 'T+00:00:00', category: 'SYSTEM',  message: 'ASTRA-HAR system initialized. Model: YOLOv8n loaded (4 classes).', step_index: 0, severity: 'info' },
  { id: 2,  timestamp: 'T+00:00:04', category: 'SYSTEM',  message: 'WebSocket uplink established. Endpoint: ws://localhost:8000/ws/detections', step_index: 0, severity: 'info' },
  { id: 3,  timestamp: 'T+00:00:10', category: 'ACTION',  message: 'Step 1 initiated: Initialize glovebox (GLOVEBOX-01)', step_index: 0, severity: 'info' },
  { id: 4,  timestamp: 'T+00:00:12', category: 'OBJECT',  message: 'GLOVEBOX-01 detected. Confidence: 0.94. Bounding box: [80,60,560,420]', step_index: 0, severity: 'info' },
  { id: 5,  timestamp: 'T+00:00:18', category: 'SYSTEM',  message: 'Nitrogen purge cycle confirmed complete. Internal pressure: 1.01 atm.', step_index: 0, severity: 'info' },
  { id: 6,  timestamp: 'T+00:00:22', category: 'ACTION',  message: 'Step 1 VALIDATED by operator CREW-01. Confidence: 0.97.', step_index: 0, severity: 'info' },
  { id: 7,  timestamp: 'T+00:01:05', category: 'ACTION',  message: 'Step 2 initiated: Insert centrifuge rotor (CENTRIFUGE-02)', step_index: 1, severity: 'info' },
  { id: 8,  timestamp: 'T+00:01:12', category: 'OBJECT',  message: 'CENTRIFUGE-02 detected. Confidence: 0.91. Rotor alignment nominal.', step_index: 1, severity: 'info' },
  { id: 9,  timestamp: 'T+00:01:38', category: 'WARNING', message: 'Rotor lock indicator amber — retry insertion detected (attempt 2/3).', step_index: 1, severity: 'warn' },
  { id: 10, timestamp: 'T+00:01:45', category: 'ACTION',  message: 'Rotor lock confirmed GREEN. Step 2 VALIDATED. Confidence: 0.93.', step_index: 1, severity: 'info' },
  { id: 11, timestamp: 'T+00:02:10', category: 'ACTION',  message: 'Step 3 initiated: Load SMP-RED into centrifuge.', step_index: 2, severity: 'info' },
  { id: 12, timestamp: 'T+00:02:14', category: 'OBJECT',  message: 'SMP-RED detected. Confidence: 0.89. Position: approaching slot A1.', step_index: 2, severity: 'info' },
  { id: 13, timestamp: 'T+00:02:31', category: 'WARNING', message: 'HOI proximity alert: hand-to-SMP distance < 2cm. Ergonomics: 94%.', step_index: 2, severity: 'warn' },
  { id: 14, timestamp: 'T+00:02:45', category: 'ACTION',  message: 'SMP-RED inserted into slot A1. Snap-lock confirmed by acoustic sensor.', step_index: 2, severity: 'info' },
  { id: 15, timestamp: 'T+00:03:02', category: 'ACTION',  message: 'Step 4 initiated: Load SMP-YEL into centrifuge.', step_index: 3, severity: 'info' },
  { id: 16, timestamp: 'T+00:03:08', category: 'OBJECT',  message: 'SMP-YEL detected. Confidence: 0.88. Mass: 14.3g — balance delta: +0.2g OK.', step_index: 3, severity: 'info' },
  { id: 17, timestamp: 'T+00:03:29', category: 'WARNING', message: 'SEQUENCE DEVIATION: SMP-YEL grasped before balance check completed.', step_index: 3, severity: 'error' },
  { id: 18, timestamp: 'T+00:03:35', category: 'ACTION',  message: 'Operator CREW-01 acknowledged deviation. Balance check completed retroactively.', step_index: 3, severity: 'warn' },
  { id: 19, timestamp: 'T+00:04:10', category: 'ACTION',  message: 'SMP-YEL inserted into slot A2. Step 4 VALIDATED. Confidence: 0.92.', step_index: 3, severity: 'info' },
  { id: 20, timestamp: 'T+00:04:30', category: 'ACTION',  message: 'Step 5 initiated: Run centrifuge cycle at 3000 RPM, 12 min.', step_index: 4, severity: 'info' },
  { id: 21, timestamp: 'T+00:04:35', category: 'SYSTEM',  message: 'Centrifuge spin-up detected. RPM ramp: 0→3000 over 45s.', step_index: 4, severity: 'info' },
  { id: 22, timestamp: 'T+00:07:12', category: 'SYSTEM',  message: 'Centrifuge at target RPM. Vibration: 0.12g (within limits <0.8g).', step_index: 4, severity: 'info' },
  { id: 23, timestamp: 'T+00:16:50', category: 'SYSTEM',  message: 'Centrifuge cycle complete. Spin-down initiated. ETA full stop: 45s.', step_index: 4, severity: 'info' },
  { id: 24, timestamp: 'T+00:17:40', category: 'ACTION',  message: 'Step 6 initiated: Extract SMP-RED sample.', step_index: 5, severity: 'info' },
  { id: 25, timestamp: 'T+00:17:52', category: 'OBJECT',  message: 'SMP-RED located in slot A1. Extraction motion detected by pose estimator.', step_index: 5, severity: 'info' },
  { id: 26, timestamp: 'T+00:18:05', category: 'ACTION',  message: 'SMP-RED successfully extracted. Step 6 VALIDATED. Confidence: 0.94.', step_index: 5, severity: 'info' },
  { id: 27, timestamp: 'T+00:18:20', category: 'ACTION',  message: 'Step 7 initiated: Document results.', step_index: 6, severity: 'info' },
  { id: 28, timestamp: 'T+00:18:45', category: 'SYSTEM',  message: 'Photographic documentation captured. File: EXP_2026_09_19_HAR_001.jpg', step_index: 6, severity: 'info' },
  { id: 29, timestamp: 'T+00:19:10', category: 'ACTION',  message: 'Experiment log entry submitted. Protocol compliance: 94.2%.', step_index: 6, severity: 'info' },
  { id: 30, timestamp: 'T+00:19:15', category: 'SYSTEM',  message: 'Telemetry downlink complete. 18.2 MB transmitted (99.2% bandwidth savings).', step_index: 6, severity: 'info' },
];

// ──────────────────────────────────────────
// SYSTEM METRICS (60 data points, 2s apart)
// ──────────────────────────────────────────
export const SYSTEM_METRICS: SystemMetric[] = Array.from({ length: 60 }, (_, i) => ({
  time: i * 2,
  cpu: 28 + Math.sin(i * 0.3) * 8 + Math.random() * 4,
  gpu: 1800 + Math.sin(i * 0.2) * 300 + Math.random() * 100,
  temp: 62 + Math.sin(i * 0.15) * 5 + Math.random() * 2,
  fps: 68 + Math.sin(i * 0.4) * 6 + Math.random() * 3,
}));