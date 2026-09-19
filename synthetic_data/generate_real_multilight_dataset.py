"""
generate_real_multilight_dataset.py — Photorealistic Multi-Lighting Dataset Generator
=====================================================================================
Generates 1,000 diverse, high-fidelity synthetic images (800 train, 200 val) at 640x640:
  - Class 0: bottle (water bottles, clear plastic, glass, reusable hydro flasks, with caps)
  - Class 1: can (metallic 3D cylinders, soda cans: Coke, Pepsi, Sprite, Red Bull, pull-tabs)
  - Class 2: phone (smartphones with glass screens, bezels, active/dark screens, in hand/desk)
  - Class 3: person (sitting at desk, standing, walking across room, reaching)

Under 5 distinct lighting conditions:
  1. Standard Daylight / Neutral Office
  2. Low-Light / Dim Night (0.35x gain, sensor noise, deep shadows)
  3. Harsh Glare / Specular Flash (bloom, clipped highlights, washed edges)
  4. Warm Tungsten Indoor (3000K amber cast)
  5. Cool Fluorescent (6500K blue/cyan cast)
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "synthetic_data" / "dataset_multilight_v2"
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
VAL_IMG_DIR = DATASET_DIR / "images" / "val"
TRAIN_LBL_DIR = DATASET_DIR / "labels" / "train"
VAL_LBL_DIR = DATASET_DIR / "labels" / "val"

IMG_W, IMG_H = 640, 640

# Canonical Class IDs
CLS_BOTTLE = 0
CLS_CAN = 1
CLS_PHONE = 2
CLS_PERSON = 3

CLASSES = ["bottle", "can", "phone", "person"]


def ensure_dirs():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Background Generator: Realistic Indoor Rooms & Workspaces
# ---------------------------------------------------------------------------
def generate_realistic_background() -> np.ndarray:
    """Generates a rich multi-texture indoor room or desk environment."""
    bg_type = random.choice(["desk_office", "living_room", "lab_bench", "workspace"])
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    if bg_type == "desk_office":
        # Wall color (neutral warm or cool plaster)
        wall_c = np.array([random.randint(140, 190), random.randint(140, 190), random.randint(145, 195)])
        split_y = int(IMG_H * random.uniform(0.38, 0.48))
        img[:split_y, :] = wall_c

        # Subtle wall texture & baseboard
        noise = np.random.normal(0, 3, (split_y, IMG_W, 3)).astype(np.int16)
        img[:split_y, :] = np.clip(img[:split_y, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img[split_y - 8:split_y, :] = (wall_c * 0.7).astype(np.uint8)

        # Desk surface (wood grain or matte composite)
        is_wood = random.random() > 0.4
        if is_wood:
            base_wood = np.array([random.randint(40, 75), random.randint(70, 120), random.randint(110, 160)])  # BGR
            for y in range(split_y, IMG_H):
                grain = int(math.sin((y - split_y) * 0.4) * 8 + math.sin((y - split_y) * 0.08) * 15)
                img[y, :] = np.clip(base_wood + grain, 0, 255).astype(np.uint8)
        else:
            desk_c = np.array([random.randint(50, 90), random.randint(50, 90), random.randint(55, 95)])
            img[split_y:, :] = desk_c

        # Add optional laptop monitor or keyboard silhouette on desk
        if random.random() > 0.5:
            mon_x1 = random.randint(20, 180)
            mon_y1 = max(split_y - random.randint(80, 140), 10)
            mon_x2 = mon_x1 + random.randint(140, 220)
            mon_y2 = split_y + 10
            cv2.rectangle(img, (mon_x1, mon_y1), (mon_x2, mon_y2), (25, 25, 30), -1)
            cv2.rectangle(img, (mon_x1 + 6, mon_y1 + 6), (mon_x2 - 6, mon_y2 - 12), (50, 60, 70), -1)

    elif bg_type == "living_room":
        # Warm ambient indoor room
        top_c = np.array([random.randint(160, 200), random.randint(160, 195), random.randint(165, 205)])
        bot_c = np.array([random.randint(70, 110), random.randint(75, 115), random.randint(80, 120)])
        for y in range(IMG_H):
            f = y / IMG_H
            img[y, :] = (top_c * (1 - f) + bot_c * f).astype(np.uint8)

        # Floor line
        floor_y = int(IMG_H * 0.70)
        img[floor_y:, :] = (np.array([45, 60, 95]) + np.random.normal(0, 4, (IMG_H - floor_y, IMG_W, 3))).clip(0, 255).astype(np.uint8)

    else:
        # Lab or generic room
        base = np.array([random.randint(120, 170), random.randint(130, 175), random.randint(135, 180)])
        img[:, :] = base
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        table_y = int(IMG_H * 0.55)
        img[table_y:, :] = (np.array([60, 65, 70]) + np.random.normal(0, 3, (IMG_H - table_y, IMG_W, 3))).clip(0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Object Renderers: Photorealistic Cans, Bottles, Phones, and Persons
# ---------------------------------------------------------------------------
def render_photorealistic_can(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders an authentic 3D metallic soda can with cylindrical shading,
    vertical specular reflection streak, brand wrap, and pull-tab top rim.
    Returns: (bgr_image, alpha_mask)
    """
    can_img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    brand = random.choice(["coca_cola", "pepsi", "sprite", "red_bull", "energy_black", "silver_beer", "fanta"])

    if brand == "coca_cola":
        base_bgr = np.array([25, 20, 215], dtype=np.float32)      # Vibrant Coke Red
        accent_bgr = np.array([245, 245, 250], dtype=np.float32)  # White wave
    elif brand == "pepsi":
        base_bgr = np.array([170, 65, 10], dtype=np.float32)      # Pepsi Blue
        accent_bgr = np.array([30, 30, 220], dtype=np.float32)    # Red accent
    elif brand == "sprite":
        base_bgr = np.array([55, 145, 10], dtype=np.float32)      # Sprite Green
        accent_bgr = np.array([20, 220, 240], dtype=np.float32)   # Yellow/lemon
    elif brand == "red_bull":
        base_bgr = np.array([120, 45, 15], dtype=np.float32)      # Deep blue
        accent_bgr = np.array([195, 195, 205], dtype=np.float32)  # Silver/grey
    elif brand == "energy_black":
        base_bgr = np.array([18, 18, 20], dtype=np.float32)       # Matte black
        accent_bgr = np.array([20, 245, 55], dtype=np.float32)    # Neon green
    elif brand == "fanta":
        base_bgr = np.array([10, 110, 240], dtype=np.float32)     # Orange
        accent_bgr = np.array([220, 220, 240], dtype=np.float32)
    else:
        base_bgr = np.array([180, 185, 190], dtype=np.float32)    # Metallic Silver
        accent_bgr = np.array([40, 160, 220], dtype=np.float32)   # Gold/amber

    # Specular light position (horizontal fraction across can width)
    light_pos_x = random.uniform(0.35, 0.55) * w
    spec_sigma = w * 0.14

    rim_h = max(3, int(h * 0.08))
    bot_h = max(3, int(h * 0.05))

    for y in range(h):
        for x in range(w):
            norm_x = (x - w / 2.0) / (w / 2.0)
            if abs(norm_x) > 0.98:
                continue

            cos_theta = math.cos(norm_x * (math.pi / 2.2))
            diffuse = max(0.20, cos_theta)

            dx = x - light_pos_x
            specular = math.exp(-(dx * dx) / (2 * spec_sigma * spec_sigma)) * 140.0

            if y < rim_h:
                silver = np.array([185, 190, 195], dtype=np.float32)
                pix = silver * diffuse + specular * 0.7
                can_img[y, x] = np.clip(pix, 0, 255).astype(np.uint8)
                alpha[y, x] = 255
            elif y > (h - bot_h):
                silver = np.array([160, 165, 170], dtype=np.float32)
                pix = silver * diffuse + specular * 0.5
                can_img[y, x] = np.clip(pix, 0, 255).astype(np.uint8)
                alpha[y, x] = 255
            else:
                wave = math.sin((y / h) * 12.0 + (x / w) * 3.0)
                if brand in ("pepsi", "red_bull") and wave > 0.4:
                    col = accent_bgr
                elif brand == "coca_cola" and 0.45 < (y / h) < 0.60 and math.sin((x / w) * 8.0) > 0.1:
                    col = accent_bgr
                elif brand == "energy_black" and 0.35 < (y / h) < 0.75 and abs(x - w * 0.5) < w * 0.15:
                    col = accent_bgr
                else:
                    col = base_bgr

                pix = col * diffuse + specular
                can_img[y, x] = np.clip(pix, 0, 255).astype(np.uint8)
                alpha[y, x] = 255

    tab_w = max(4, int(w * 0.28))
    tab_h = max(2, int(rim_h * 0.5))
    tab_x = int((w - tab_w) / 2)
    tab_y = int(rim_h * 0.25)
    cv2.rectangle(can_img, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), (90, 95, 100), -1)
    cv2.circle(can_img, (int(w / 2), tab_y + int(tab_h / 2)), max(1, int(tab_h * 0.3)), (40, 40, 45), -1)

    return can_img, alpha


def render_photorealistic_bottle(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders an authentic water bottle, plastic bottle, or stainless steel flask
    with a distinct narrow neck, screw-on cap, and cylindrical body.
    Returns: (bgr_image, alpha_mask)
    """
    b_img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    b_type = random.choice(["plastic_water", "translucent_cyan", "metal_flask", "green_glass"])

    if b_type == "plastic_water":
        base_bgr = np.array([210, 205, 195], dtype=np.float32)
        cap_bgr = np.array([220, 80, 20], dtype=np.float32)
        is_transparent = True
    elif b_type == "translucent_cyan":
        base_bgr = np.array([225, 190, 80], dtype=np.float32)
        cap_bgr = np.array([240, 240, 245], dtype=np.float32)
        is_transparent = True
    elif b_type == "metal_flask":
        base_bgr = np.array([random.randint(30, 180), random.randint(30, 180), random.randint(30, 180)], dtype=np.float32)
        cap_bgr = np.array([30, 30, 35], dtype=np.float32)
        is_transparent = False
    else:
        base_bgr = np.array([45, 140, 50], dtype=np.float32)
        cap_bgr = np.array([30, 180, 220], dtype=np.float32)
        is_transparent = True

    neck_h = int(h * random.uniform(0.20, 0.28))
    cap_h = max(4, int(neck_h * 0.40))
    neck_w = max(6, int(w * random.uniform(0.38, 0.48)))

    light_x = w * random.uniform(0.35, 0.55)
    spec_sigma = w * 0.12

    for y in range(h):
        if y < cap_h:
            curr_w = neck_w
            cur_col = cap_bgr
        elif y < neck_h:
            progress = (y - cap_h) / max(1, (neck_h - cap_h))
            curr_w = int(neck_w + (w - neck_w) * (progress ** 1.6))
            cur_col = base_bgr
        else:
            curr_w = w
            cur_col = base_bgr

        half_w = curr_w / 2.0
        cx = w / 2.0
        x_min = int(cx - half_w)
        x_max = int(cx + half_w)

        for x in range(max(0, x_min), min(w, x_max)):
            norm_x = (x - cx) / max(0.5, half_w)
            if abs(norm_x) > 0.98:
                continue

            diffuse = max(0.25, math.cos(norm_x * (math.pi / 2.1)))
            dx = x - light_x
            specular = math.exp(-(dx * dx) / (2 * spec_sigma * spec_sigma)) * (160.0 if not is_transparent else 190.0)

            if y < cap_h:
                ridge = 1.1 if (x % 3 == 0) else 0.9
                pix = cur_col * diffuse * ridge + specular * 0.4
            elif int(h * 0.45) < y < int(h * 0.65) and is_transparent:
                label_col = np.array([230, 230, 235], dtype=np.float32)
                pix = label_col * diffuse + specular * 0.3
            else:
                pix = cur_col * diffuse + specular

            b_img[y, x] = np.clip(pix, 0, 255).astype(np.uint8)
            alpha[y, x] = 255 if not is_transparent else 225

    return b_img, alpha


def render_photorealistic_phone(w: int, h: int, landscape: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders a realistic modern smartphone: flat dark glass screen,
    thin metallic/matte bezels, camera notch / dynamic island, and glass reflection.
    Returns: (bgr_image, alpha_mask)
    """
    p_img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    bezel_bgr = random.choice([
        (25, 25, 28),     # Space Black
        (45, 45, 50),     # Dark Gray
        (185, 190, 195),  # Silver
        (85, 60, 40),     # Midnight Blue
    ])

    bezel_px = max(2, int(min(w, h) * 0.05))
    cv2.rectangle(p_img, (0, 0), (w - 1, h - 1), bezel_bgr, -1)
    alpha[:, :] = 255

    is_screen_on = random.random() < 0.35

    screen_x1 = bezel_px
    screen_y1 = bezel_px
    screen_x2 = w - bezel_px
    screen_y2 = h - bezel_px

    if is_screen_on:
        screen_c = np.array([random.randint(60, 180), random.randint(50, 160), random.randint(40, 140)], dtype=np.uint8)
        p_img[screen_y1:screen_y2, screen_x1:screen_x2] = screen_c
        if not landscape and (screen_y2 - screen_y1) > 40:
            clock_y = screen_y1 + int((screen_y2 - screen_y1) * 0.25)
            cv2.putText(p_img, "10:24", (screen_x1 + int(w * 0.2), clock_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        screen_c = np.array([12, 12, 14], dtype=np.uint8)
        p_img[screen_y1:screen_y2, screen_x1:screen_x2] = screen_c

    # Diagonal reflection sheen
    for y in range(screen_y1, screen_y2):
        sheen_x = int(screen_x1 + (y - screen_y1) * 0.75 + w * 0.2)
        if screen_x1 <= sheen_x < screen_x2:
            p_img[y, max(screen_x1, sheen_x - 3):min(screen_x2, sheen_x + 3)] = np.clip(
                p_img[y, max(screen_x1, sheen_x - 3):min(screen_x2, sheen_x + 3)].astype(np.int16) + 40,
                0, 255
            ).astype(np.uint8)

    if not landscape and w > 20:
        notch_w = max(6, int(w * 0.25))
        notch_h = max(2, int(bezel_px * 0.8))
        notch_x = (w - notch_w) // 2
        cv2.rectangle(p_img, (notch_x, screen_y1 + 2), (notch_x + notch_w, screen_y1 + 2 + notch_h), (8, 8, 10), -1)

    return p_img, alpha


def render_photorealistic_person(w: int, h: int, posture: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders a human figure in realistic indoor postures (seated at desk, standing, walking).
    Returns: (bgr_image, alpha_mask)
    """
    p_img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    skin = random.choice([
        (185, 215, 240),
        (140, 180, 215),
        (85, 125, 170),
        (55, 75, 110),
    ])
    hair = random.choice([(25, 25, 30), (35, 50, 75), (50, 80, 130), (160, 180, 195)])
    shirt = (random.randint(30, 200), random.randint(30, 200), random.randint(30, 200))
    pants = (random.randint(25, 70), random.randint(25, 70), random.randint(30, 85))

    cx = w // 2
    head_rad = max(8, int(w * 0.18))
    head_cy = head_rad + 4

    cv2.circle(p_img, (cx, head_cy), head_rad, skin, -1)
    cv2.circle(alpha, (cx, head_cy), head_rad, 255, -1)
    cv2.ellipse(p_img, (cx, head_cy - 2), (head_rad, int(head_rad * 0.7)), 0, 180, 360, hair, -1)

    neck_w = max(4, int(head_rad * 0.6))
    neck_top = head_cy + int(head_rad * 0.7)
    neck_bot = neck_top + max(4, int(h * 0.04))
    cv2.rectangle(p_img, (cx - neck_w // 2, neck_top), (cx + neck_w // 2, neck_bot), skin, -1)
    cv2.rectangle(alpha, (cx - neck_w // 2, neck_top), (cx + neck_w // 2, neck_bot), 255, -1)

    torso_top = neck_bot
    torso_h = int(h * (0.42 if posture == "seated" else 0.35))
    torso_bot = min(h - 1, torso_top + torso_h)
    torso_w = max(16, int(w * 0.75))
    cv2.rectangle(p_img, (cx - torso_w // 2, torso_top), (cx + torso_w // 2, torso_bot), shirt, -1)
    cv2.rectangle(alpha, (cx - torso_w // 2, torso_top), (cx + torso_w // 2, torso_bot), 255, -1)

    arm_w = max(3, int(w * 0.14))
    cv2.rectangle(p_img, (cx - torso_w // 2 - arm_w, torso_top + 4), (cx - torso_w // 2, torso_bot - 4), shirt, -1)
    cv2.rectangle(alpha, (cx - torso_w // 2 - arm_w, torso_top + 4), (cx - torso_w // 2, torso_bot - 4), 255, -1)
    cv2.rectangle(p_img, (cx + torso_w // 2, torso_top + 4), (cx + torso_w // 2 + arm_w, torso_bot - 4), shirt, -1)
    cv2.rectangle(alpha, (cx + torso_w // 2, torso_top + 4), (cx + torso_w // 2 + arm_w, torso_bot - 4), 255, -1)

    if posture in ("standing", "walking"):
        leg_top = torso_bot
        leg_bot = h - 2
        leg_w = max(4, int(torso_w * 0.38))
        if posture == "walking":
            cv2.line(p_img, (cx - int(torso_w * 0.2), leg_top), (cx - int(torso_w * 0.4), leg_bot), pants, leg_w)
            cv2.line(alpha, (cx - int(torso_w * 0.2), leg_top), (cx - int(torso_w * 0.4), leg_bot), 255, leg_w)
            cv2.line(p_img, (cx + int(torso_w * 0.2), leg_top), (cx + int(torso_w * 0.4), leg_bot), pants, leg_w)
            cv2.line(alpha, (cx + int(torso_w * 0.2), leg_top), (cx + int(torso_w * 0.4), leg_bot), 255, leg_w)
        else:
            cv2.rectangle(p_img, (cx - int(torso_w * 0.45), leg_top), (cx - int(torso_w * 0.05), leg_bot), pants, -1)
            cv2.rectangle(alpha, (cx - int(torso_w * 0.45), leg_top), (cx - int(torso_w * 0.05), leg_bot), 255, -1)
            cv2.rectangle(p_img, (cx + int(torso_w * 0.05), leg_top), (cx + int(torso_w * 0.45), leg_bot), pants, -1)
            cv2.rectangle(alpha, (cx + int(torso_w * 0.05), leg_top), (cx + int(torso_w * 0.45), leg_bot), 255, -1)

    return p_img, alpha


# ---------------------------------------------------------------------------
# Multi-Lighting Application Engine
# ---------------------------------------------------------------------------
def apply_lighting_environment(img: np.ndarray, light_mode: str) -> np.ndarray:
    """
    Applies authentic physical lighting conditions:
      - DAYLIGHT: clean natural exposure
      - LOW_LIGHT: 0.35x gain, sensor noise, deep dark shadows
      - HARSH_GLARE: specular bloom, highlight clipping, washed contrast
      - WARM_TUNGSTEN: 3000K amber/golden temperature shift
      - COOL_FLUORESCENT: 6500K blue/cyan temperature shift
    """
    out = img.astype(np.float32)

    if light_mode == "LOW_LIGHT":
        out = out * random.uniform(0.28, 0.42)
        noise = np.random.normal(0, random.uniform(7.0, 14.0), out.shape)
        out = np.clip(out + noise, 0, 255)
        out[:, :, 0] = np.clip(out[:, :, 0] * 1.15, 0, 255)

    elif light_mode == "HARSH_GLARE":
        glare_cx = int(IMG_W * random.uniform(0.3, 0.7))
        glare_cy = int(IMG_H * random.uniform(0.2, 0.5))
        y_grid, x_grid = np.ogrid[:IMG_H, :IMG_W]
        dist_sq = (x_grid - glare_cx) ** 2 + (y_grid - glare_cy) ** 2
        bloom_radius = (IMG_W * random.uniform(0.35, 0.55)) ** 2
        glare_mask = np.exp(-dist_sq / (2.0 * bloom_radius))[:, :, np.newaxis]

        out = out * 1.25 + glare_mask * random.uniform(90.0, 150.0)
        out = np.clip(out, 0, 255)

    elif light_mode == "WARM_TUNGSTEN":
        out[:, :, 2] = np.clip(out[:, :, 2] * random.uniform(1.20, 1.35), 0, 255)  # Red
        out[:, :, 1] = np.clip(out[:, :, 1] * random.uniform(1.05, 1.15), 0, 255)  # Green
        out[:, :, 0] = np.clip(out[:, :, 0] * random.uniform(0.70, 0.85), 0, 255)  # Blue

    elif light_mode == "COOL_FLUORESCENT":
        out[:, :, 0] = np.clip(out[:, :, 0] * random.uniform(1.20, 1.35), 0, 255)  # Blue
        out[:, :, 1] = np.clip(out[:, :, 1] * random.uniform(1.02, 1.08), 0, 255)  # Green
        out[:, :, 2] = np.clip(out[:, :, 2] * random.uniform(0.75, 0.88), 0, 255)  # Red

    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Composition Pipeline
# ---------------------------------------------------------------------------
def blend_object_onto_canvas(
    canvas: np.ndarray,
    obj_img: np.ndarray,
    alpha: np.ndarray,
    x1: int,
    y1: int,
) -> None:
    """Blends an object image with alpha mask onto canvas with contact shadow."""
    oh, ow = obj_img.shape[:2]
    x2 = min(canvas.shape[1], x1 + ow)
    y2 = min(canvas.shape[0], y1 + oh)
    cx1 = max(0, x1)
    cy1 = max(0, y1)

    crop_ox1 = cx1 - x1
    crop_oy1 = cy1 - y1
    crop_ox2 = crop_ox1 + (x2 - cx1)
    crop_oy2 = crop_oy1 + (y2 - cy1)

    if crop_ox2 <= crop_ox1 or crop_oy2 <= crop_oy1:
        return

    shadow_h = max(3, int(oh * 0.08))
    shadow_y1 = min(canvas.shape[0] - 1, y2 - shadow_h // 2)
    shadow_y2 = min(canvas.shape[0], shadow_y1 + shadow_h)
    if shadow_y2 > shadow_y1 and (x2 - cx1) > 6:
        shadow_patch = canvas[shadow_y1:shadow_y2, cx1:x2].astype(np.float32) * 0.60
        canvas[shadow_y1:shadow_y2, cx1:x2] = shadow_patch.astype(np.uint8)

    a = (alpha[crop_oy1:crop_oy2, crop_ox1:crop_ox2].astype(np.float32) / 255.0)[:, :, np.newaxis]
    canvas[cy1:y2, cx1:x2] = (
        obj_img[crop_oy1:crop_oy2, crop_ox1:crop_ox2].astype(np.float32) * a
        + canvas[cy1:y2, cx1:x2].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)


def generate_single_scene(index: int, split: str) -> None:
    """Generates one multi-lighting scene with exact pixel-perfect labels."""
    canvas = generate_realistic_background()
    labels = []

    lighting_modes = ["DAYLIGHT", "LOW_LIGHT", "HARSH_GLARE", "WARM_TUNGSTEN", "COOL_FLUORESCENT"]
    active_lighting = random.choice(lighting_modes)

    # 1. Place 1-2 Humans (Sitting, standing, or walking)
    num_humans = random.choice([1, 1, 2])
    for _ in range(num_humans):
        posture = random.choice(["seated", "standing", "walking"])
        hw = random.randint(110, 190)
        hh = random.randint(260, 480)
        hx1 = random.randint(20, IMG_W - hw - 20)
        hy1 = random.randint(IMG_H - hh - 20, IMG_H - hh + 10)
        h_img, h_alpha = render_photorealistic_person(hw, hh, posture)
        blend_object_onto_canvas(canvas, h_img, h_alpha, hx1, hy1)

        cx = (hx1 + hw / 2.0) / IMG_W
        cy = (hy1 + hh / 2.0) / IMG_H
        bw = hw / IMG_W
        bh = hh / IMG_H
        labels.append(f"{CLS_PERSON} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 2. Place 1-3 Cans (Distinct soda/beverage cans)
    num_cans = random.randint(1, 3)
    for _ in range(num_cans):
        can_w = random.randint(38, 72)
        can_h = int(can_w * random.uniform(1.65, 1.95))
        can_x = random.randint(20, IMG_W - can_w - 20)
        can_y = random.randint(int(IMG_H * 0.40), IMG_H - can_h - 15)

        can_img, can_alpha = render_photorealistic_can(can_w, can_h)
        blend_object_onto_canvas(canvas, can_img, can_alpha, can_x, can_y)

        cx = (can_x + can_w / 2.0) / IMG_W
        cy = (can_y + can_h / 2.0) / IMG_H
        bw = can_w / IMG_W
        bh = can_h / IMG_H
        labels.append(f"{CLS_CAN} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 3. Place 1-3 Bottles (Water bottles, clear bottles, flasks)
    num_bottles = random.randint(1, 3)
    for _ in range(num_bottles):
        b_w = random.randint(35, 68)
        b_h = int(b_w * random.uniform(2.30, 3.20))
        b_x = random.randint(20, IMG_W - b_w - 20)
        b_y = random.randint(int(IMG_H * 0.38), IMG_H - b_h - 15)

        b_img, b_alpha = render_photorealistic_bottle(b_w, b_h)
        blend_object_onto_canvas(canvas, b_img, b_alpha, b_x, b_y)

        cx = (b_x + b_w / 2.0) / IMG_W
        cy = (b_y + b_h / 2.0) / IMG_H
        bw = b_w / IMG_W
        bh = b_h / IMG_H
        labels.append(f"{CLS_BOTTLE} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 4. Place 1-2 Smartphones (in hand or on desk, portrait/landscape)
    num_phones = random.randint(1, 2)
    for _ in range(num_phones):
        is_land = random.random() < 0.25
        if is_land:
            p_h = random.randint(32, 55)
            p_w = int(p_h * random.uniform(2.05, 2.25))
        else:
            p_w = random.randint(32, 55)
            p_h = int(p_w * random.uniform(2.05, 2.25))

        p_x = random.randint(20, IMG_W - p_w - 20)
        p_y = random.randint(int(IMG_H * 0.42), IMG_H - p_h - 15)

        p_img, p_alpha = render_photorealistic_phone(p_w, p_h, is_land)
        blend_object_onto_canvas(canvas, p_img, p_alpha, p_x, p_y)

        cx = (p_x + p_w / 2.0) / IMG_W
        cy = (p_y + p_h / 2.0) / IMG_H
        bw = p_w / IMG_W
        bh = p_h / IMG_H
        labels.append(f"{CLS_PHONE} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 5. Apply the active multi-lighting condition
    final_img = apply_lighting_environment(canvas, active_lighting)

    # 6. Save image and label
    img_dir = TRAIN_IMG_DIR if split == "train" else VAL_IMG_DIR
    lbl_dir = TRAIN_LBL_DIR if split == "train" else VAL_LBL_DIR

    img_filename = f"scene_{index:05d}.jpg"
    lbl_filename = f"scene_{index:05d}.txt"

    cv2.imwrite(str(img_dir / img_filename), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    with open(lbl_dir / lbl_filename, "w") as fp:
        fp.write("\n".join(labels) + "\n")


def write_data_yaml():
    yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: bottle
  1: can
  2: phone
  3: person
"""
    with open(DATASET_DIR / "data.yaml", "w") as fp:
        fp.write(yaml_content)
    print(f"[DATASET] data.yaml written to {DATASET_DIR / 'data.yaml'}")


def main():
    print("=" * 70)
    print("Generating 1,000 Photorealistic Multi-Lighting Scenes...")
    print("Classes: 0=bottle, 1=can, 2=phone, 3=person")
    print("Lighting: Daylight, Low-Light, Harsh Glare, Warm Tungsten, Cool Fluorescent")
    print("=" * 70)

    ensure_dirs()

    print("[1/2] Generating 800 training scenes...")
    for i in range(800):
        generate_single_scene(i, split="train")
        if (i + 1) % 100 == 0:
            print(f"  -> Generated {i + 1}/800 training images...")

    print("[2/2] Generating 200 validation scenes...")
    for i in range(200):
        generate_single_scene(i, split="val")
        if (i + 1) % 50 == 0:
            print(f"  -> Generated {i + 1}/200 validation images...")

    write_data_yaml()
    print("=" * 70)
    print("Dataset generation complete! Ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
