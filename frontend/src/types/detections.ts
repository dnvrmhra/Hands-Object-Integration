export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface Attitude3D {
  roll: number;
  pitch: number;
  yaw: number;
  is_invariant?: boolean;
}

export interface HOISummary {
  state: 'IDLE' | 'REACHING' | 'GRASPING' | 'TRANSPORTING' | 'RELEASED';
  target_object: string | null;
  confidence: number;
  proximity_cm: number;
  action_label: string;
  attitude_3d?: Attitude3D;
  motion_correlation?: number;
  hand_speed_px?: number;
  object_speed_px?: number;
  is_grasping?: boolean;
  pinch_distance_cm?: number;
}

export interface HandData {
  hand_id: number;
  handedness: string;
  palm_px: [number, number];
  wrist_px: [number, number];
  is_grasping: boolean;
  pinch_distance_px: number;
  attitude_3d?: Attitude3D;
  velocity: [number, number];
  speed: number;
}

export interface MovementEvent {
  id: number;
  object_name: string;
  start_pos: [number, number];
  end_pos: [number, number];
  distance_cm: number;
  duration_s: number;
  timestamp: string;
}

export interface MovementTelemetry {
  state: 'AT_REST' | 'GRASPED' | 'TRANSPORTING' | 'PLACED';
  active_object: string | null;
  start_point?: [number, number];
  current_point?: [number, number];
  end_point?: [number, number];
  distance_cm: number;
  speed_cm_s: number;
  is_placed_recently: boolean;
  history: MovementEvent[];
}

export interface HumanData {
  track_id: number;
  class_id?: number;
  class_name: string;
  state: 'IDLE' | 'MOVING' | 'WALKING';
  speed_px_s: number;
  displacement_px: number;
  velocity: [number, number];
  confidence: number;
  bbox: BoundingBox;
}

export interface DetectedCounts {
  bottles: number;
  cans: number;
  phones: number;
  humans: number;
  human_state: string;
}

export interface DetectionFrame {
  frame_id: number;
  timestamp: number;
  fps: number;
  detections: Detection[];
  bandwidth_saved_pct: number;
  active_step: number;
  hoi?: HOISummary;
  hands?: HandData[];
  humans?: HumanData[];
  attitude_3d?: Attitude3D;
  movement?: MovementTelemetry;
  camera_source?: string;
  lighting?: string;
  counts?: DetectedCounts;
}

export interface ProcessedVideoData {
  status: string;
  video_url: string;
  filename: string;
  total_frames: number;
  duration_sec: number;
  action_counts: Record<string, number>;
  frames: DetectionFrame[];
}

export interface ProtocolStep {
  id: number;
  action: string;
  object: string;
  status: 'completed' | 'active' | 'pending';
  confidence: number;
  timestamp?: string;
  guidance: string;
}

export interface AuditEvent {
  id: number;
  timestamp: string;
  category: 'ACTION' | 'OBJECT' | 'WARNING' | 'SYSTEM';
  message: string;
  step_index: number;
  severity?: 'info' | 'warn' | 'error';
}

export interface SystemMetric {
  time: number;
  cpu: number;
  gpu: number;
  temp: number;
  fps: number;
}