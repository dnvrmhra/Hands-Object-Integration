"""
generate_multilight_dataset.py — Multi-Lighting Dataset Generator
==================================================================
Generates 1,000 diverse synthetic scenes for:
  - Class 0: bottle
  - Class 1: can
  - Class 2: phone
  - Class 3: person (idle, moving, walking)

Under 5 distinct lighting conditions:
  1. Standard Daylight / Office
  2. Low-Light / Dim Night Mode (dark shadows, sensor grain, underexposed)
  3. Harsh Glare / Overexposed / Backlit (specular highlights, bloom, lens flare)
  4. Warm Tungsten Indoor (amber/yellow 3000K cast)
  5. Cool Fluorescent (cyan/blue 6500K cast)
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "synthetic_data" / "dataset_multilight"
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
VAL_IMG_DIR = DATASET_DIR / "images" / "val"
TRAIN_LBL_DIR = DATASET_DIR / "labels" / "train"
VAL_LBL_DIR = DATASET_DIR / "labels" / "val"

IMG_W, IMG_H = 640, 640

CLASSES = ["bottle", "can", "phone", "person"]


def ensure_dirs():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def draw_background() -> Image.Image:
    """Generates an indoor office/room background with desk or wall lines."""
    bg_type = random.choice(["office", "desk", "room", "lab"])
    img = Image.new("RGB", (IMG_W, IMG_H))
    draw = ImageDraw.Draw(img)

    if bg_type == "desk":
        wall_color = (random.randint(60, 140), random.randint(70, 150), random.randint(80, 160))
        desk_color = (random.randint(40, 100), random.randint(35, 80), random.randint(30, 70))
        draw.rectangle([0, 0, IMG_W, int(IMG_H * 0.45)], fill=wall_color)
        draw.rectangle([0, int(IMG_H * 0.45), IMG_W, IMG_H], fill=desk_color)
        draw.line([0, int(IMG_H * 0.45), IMG_W, int(IMG_H * 0.45)], fill=(30, 30, 30), width=3)
    elif bg_type == "office":
        base = (random.randint(100, 160), random.randint(100, 160), random.randint(110, 170))
        floor = (random.randint(70, 110), random.randint(70, 110), random.randint(70, 110))
        draw.rectangle([0, 0, IMG_W, int(IMG_H * 0.6)], fill=base)
        draw.rectangle([0, int(IMG_H * 0.6), IMG_W, IMG_H], fill=floor)
        # Add door / whiteboard outline
        draw.rectangle([int(IMG_W * 0.1), int(IMG_H * 0.1), int(IMG_W * 0.4), int(IMG_H * 0.5)], outline=(80, 80, 90), width=2)
    else:
        # Gradient background
        top_c = np.array([random.randint(40, 120), random.randint(40, 120), random.randint(50, 130)])
        bot_c = np.array([random.randint(20, 60), random.randint(20, 60), random.randint(25, 70)])
        arr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        for y in range(IMG_H):
            factor = y / IMG_H
            arr[y, :] = (top_c * (1 - factor) + bot_c * factor).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def draw_person(img: Image.Image, posture: str) -> tuple[float, float, float, float]:
    """Draws a human figure (standing, walking, sitting) and returns normalized YOLO bbox."""
    draw = ImageDraw.Draw(img)
    w = random.randint(90, 180)
    h = random.randint(240, 480)
    x1 = random.randint(20, IMG_W - w - 20)
    y1 = random.randint(IMG_H - h - 30, IMG_H - h + 10)
    x2 = x1 + w
    y2 = min(IMG_H - 5, y1 + h)

    skin_color = random.choice([(235, 195, 170), (200, 150, 120), (160, 110, 80), (110, 75, 55)])
    shirt_color = (random.randint(30, 220), random.randint(30, 220), random.randint(30, 220))
    pants_color = (random.randint(20, 80), random.randint(20, 80), random.randint(40, 110))

    head_rad = w // 3.5
    head_cx = (x1 + x2) // 2
    head_cy = y1 + head_rad

    # Head
    draw.ellipse([head_cx - head_rad, head_cy - head_rad, head_cx + head_rad, head_cy + head_rad], fill=skin_color)
    # Hair
    hair_color = random.choice([(30, 20, 15), (60, 40, 25), (160, 130, 80), (20, 20, 20)])
    draw.arc([head_cx - head_rad, head_cy - head_rad, head_cx + head_rad, head_cy], 180, 360, fill=hair_color, width=4)

    # Torso
    torso_top = int(head_cy + head_rad * 0.8)
    torso_bottom = int(torso_top + (y2 - torso_top) * 0.45)
    draw.rectangle([x1 + w // 6, torso_top, x2 - w // 6, torso_bottom], fill=shirt_color)

    # Arms
    arm_w = w // 7
    draw.rectangle([x1, torso_top, x1 + arm_w, torso_bottom + 20], fill=shirt_color)
    draw.rectangle([x2 - arm_w, torso_top, x2, torso_bottom + 20], fill=shirt_color)
    # Hands
    draw.ellipse([x1, torso_bottom + 15, x1 + arm_w + 4, torso_bottom + 30], fill=skin_color)
    draw.ellipse([x2 - arm_w - 4, torso_bottom + 15, x2, torso_bottom + 30], fill=skin_color)

    # Legs (walking stride vs standing still)
    leg_top = torso_bottom
    leg_bottom = y2
    if posture == "walking":
        # Stride: left leg forward, right leg back
        stride_offset = w // 4
        draw.polygon([(x1 + w // 5, leg_top), (x1 + w // 5 - stride_offset, leg_bottom),
                      (x1 + w // 2 - stride_offset, leg_bottom), (x1 + w // 2, leg_top)], fill=pants_color)
        draw.polygon([(x1 + w // 2, leg_top), (x1 + w // 2 + stride_offset, leg_bottom),
                      (x2 - w // 5 + stride_offset, leg_bottom), (x2 - w // 5, leg_top)], fill=pants_color)
    else:
        # Standing / Idle
        draw.rectangle([x1 + w // 5, leg_top, x1 + w // 2 - 2, leg_bottom], fill=pants_color)
        draw.rectangle([x1 + w // 2 + 2, leg_top, x2 - w // 5, leg_bottom], fill=pants_color)

    # Convert to normalized YOLO format (cx, cy, w, h)
    cx = (x1 + x2) / (2.0 * IMG_W)
    cy = (y1 + y2) / (2.0 * IMG_H)
    nw = (x2 - x1) / IMG_W
    nh = (y2 - y1) / IMG_H
    return cx, cy, nw, nh


def draw_bottle(img: Image.Image) -> tuple[float, float, float, float]:
    """Draws a water bottle or flask and returns normalized YOLO bbox."""
    draw = ImageDraw.Draw(img)
    w = random.randint(28, 65)
    h = int(w * random.uniform(2.2, 3.8))
    x1 = random.randint(15, IMG_W - w - 15)
    y1 = random.randint(50, IMG_H - h - 15)
    x2 = x1 + w
    y2 = y1 + h

    bottle_color = random.choice([
        (50, 150, 220),   # Cyan plastic
        (220, 220, 230),   # Silver/stainless steel
        (40, 160, 120),   # Emerald green
        (200, 70, 70),     # Red flask
        (70, 70, 80),      # Dark grey
    ])
    cap_color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))

    # Bottle body
    draw.rounded_rectangle([x1, y1 + int(h * 0.2), x2, y2], radius=6, fill=bottle_color)
    # Neck
    neck_w = w // 2
    neck_x1 = x1 + (w - neck_w) // 2
    draw.rectangle([neck_x1, y1 + int(h * 0.08), neck_x1 + neck_w, y1 + int(h * 0.22)], fill=bottle_color)
    # Cap
    draw.rectangle([neck_x1 - 2, y1, neck_x1 + neck_w + 2, y1 + int(h * 0.08)], fill=cap_color)
    # Highlight reflection line
    draw.line([x1 + 4, y1 + int(h * 0.25), x1 + 4, y2 - 8], fill=(255, 255, 255, 180), width=2)

    cx = (x1 + x2) / (2.0 * IMG_W)
    cy = (y1 + y2) / (2.0 * IMG_H)
    return cx, cy, (x2 - x1) / IMG_W, (y2 - y1) / IMG_H


def draw_can(img: Image.Image) -> tuple[float, float, float, float]:
    """Draws a soda/beverage can and returns normalized YOLO bbox."""
    draw = ImageDraw.Draw(img)
    w = random.randint(30, 60)
    h = int(w * random.uniform(1.4, 2.0))  # Cans have aspect ratio ~ 1.5 - 2.0
    x1 = random.randint(15, IMG_W - w - 15)
    y1 = random.randint(50, IMG_H - h - 15)
    x2 = x1 + w
    y2 = y1 + h

    can_color = random.choice([
        (220, 30, 30),    # Coke red
        (30, 70, 190),    # Pepsi blue
        (220, 200, 40),   # Lemon yellow
        (40, 180, 60),    # Sprite green
        (190, 190, 200),  # Silver Diet / aluminium
        (30, 30, 35),     # Monster black
    ])
    rim_color = (200, 200, 210)

    # Can body
    draw.rounded_rectangle([x1, y1, x2, y2], radius=4, fill=can_color)
    # Top and bottom silver rims
    draw.ellipse([x1, y1, x2, y1 + int(h * 0.12)], fill=rim_color)
    draw.ellipse([x1, y2 - int(h * 0.10), x2, y2], fill=rim_color)
    # Top pull-tab oval
    tab_w = w // 3
    draw.ellipse([x1 + w // 3, y1 + 2, x1 + w // 3 + tab_w, y1 + int(h * 0.08)], fill=(120, 120, 130))
    # Vertical specular shine
    draw.line([x1 + int(w * 0.25), y1 + 5, x1 + int(w * 0.25), y2 - 5], fill=(255, 255, 255, 120), width=3)

    cx = (x1 + x2) / (2.0 * IMG_W)
    cy = (y1 + y2) / (2.0 * IMG_H)
    return cx, cy, (x2 - x1) / IMG_W, (y2 - y1) / IMG_H


def draw_phone(img: Image.Image) -> tuple[float, float, float, float]:
    """Draws a smartphone and returns normalized YOLO bbox."""
    draw = ImageDraw.Draw(img)
    w = random.randint(30, 55)
    h = int(w * random.uniform(1.8, 2.3))  # Phone aspect ratio ~ 2.0
    x1 = random.randint(15, IMG_W - w - 15)
    y1 = random.randint(50, IMG_H - h - 15)
    x2 = x1 + w
    y2 = y1 + h

    body_color = random.choice([(25, 25, 30), (50, 50, 55), (210, 210, 220), (30, 45, 60)])
    screen_color = (random.randint(15, 35), random.randint(20, 45), random.randint(35, 65))

    # Outer bezel with rounded corners
    draw.rounded_rectangle([x1, y1, x2, y2], radius=5, fill=body_color)
    # Screen
    margin = 3
    draw.rounded_rectangle([x1 + margin, y1 + margin + 3, x2 - margin, y2 - margin - 3], radius=3, fill=screen_color)
    # Camera bump / punch-hole camera
    cam_cx = (x1 + x2) // 2
    draw.ellipse([cam_cx - 2, y1 + 2, cam_cx + 2, y1 + 6], fill=(10, 10, 15))
    # Screen glare diagonal line
    draw.line([x1 + 6, y1 + 10, x2 - 6, y2 - 10], fill=(255, 255, 255, 60), width=1)

    cx = (x1 + x2) / (2.0 * IMG_W)
    cy = (y1 + y2) / (2.0 * IMG_H)
    return cx, cy, (x2 - x1) / IMG_W, (y2 - y1) / IMG_H


def apply_lighting_transform(img: Image.Image, light_mode: str) -> Image.Image:
    """Applies realistic lighting transformations to the entire synthesized scene."""
    if light_mode == "low_light":
        # 1. Low-Light: Dark exposure, sensor grain, high shadows
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.28, 0.48))
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(random.uniform(0.85, 1.25))

        # Add Gaussian sensor noise
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, random.uniform(8, 18), arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif light_mode == "glare":
        # 2. Harsh Glare: High exposure, specular bloom, lens wash-out
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(1.45, 1.95))
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(random.uniform(1.15, 1.45))

        # Add circular glare bloom
        bloom = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
        bloom_draw = ImageDraw.Draw(bloom)
        gx = random.randint(int(IMG_W * 0.2), int(IMG_W * 0.8))
        gy = random.randint(int(IMG_H * 0.1), int(IMG_H * 0.6))
        gr = random.randint(70, 160)
        bloom_draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(255, 255, 240, random.randint(70, 140)))
        bloom = bloom.filter(ImageFilter.GaussianBlur(radius=random.randint(15, 30)))
        img = Image.alpha_composite(img.convert("RGBA"), bloom).convert("RGB")

    elif light_mode == "tungsten_warm":
        # 3. Warm Tungsten Indoor: Rich amber/yellow cast (3000K)
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.25 + 15, 0, 255)  # Boost Red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.08 + 5, 0, 255)   # Slight Green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.72 - 10, 0, 255)  # Reduce Blue
        img = Image.fromarray(arr.astype(np.uint8))

    elif light_mode == "fluorescent_cool":
        # 4. Cool Fluorescent: Cyan/blue cast (6500K)
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.80 - 10, 0, 255)  # Reduce Red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.28 + 20, 0, 255)  # Boost Blue
        img = Image.fromarray(arr.astype(np.uint8))

    elif light_mode == "cast_shadow":
        # 5. Directional Harsh Cast Shadow across half the image
        shadow_mask = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_mask)
        sx = random.randint(int(IMG_W * 0.3), int(IMG_W * 0.7))
        sdraw.polygon([(sx, 0), (IMG_W, 0), (IMG_W, IMG_H), (sx - 100, IMG_H)], fill=(0, 0, 0, random.randint(90, 160)))
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=8))
        img = Image.alpha_composite(img.convert("RGBA"), shadow_mask).convert("RGB")

    return img


def generate_single_sample(index: int, split: str) -> None:
    """Renders an image with random combination of humans, cans, bottles, phones under varied lighting."""
    img = draw_background()
    labels: list[str] = []

    # Lighting mode rotation
    lighting_modes = ["standard", "low_light", "glare", "tungsten_warm", "fluorescent_cool", "cast_shadow"]
    light_mode = lighting_modes[index % len(lighting_modes)]

    # 1. Decide human presence (85% chance)
    if random.random() < 0.85:
        posture = random.choice(["idle", "moving", "walking"])
        cx, cy, nw, nh = draw_person(img, posture)
        labels.append(f"3 {cx:.5f} {cy:.5f} {nw:.5f} {nh:.5f}")

    # 2. Decide cans (1 to 2 cans, 70% chance)
    if random.random() < 0.75:
        num_cans = random.randint(1, 2)
        for _ in range(num_cans):
            cx, cy, nw, nh = draw_can(img)
            labels.append(f"1 {cx:.5f} {cy:.5f} {nw:.5f} {nh:.5f}")

    # 3. Decide bottles (1 to 2 bottles, 75% chance)
    if random.random() < 0.75:
        num_bottles = random.randint(1, 2)
        for _ in range(num_bottles):
            cx, cy, nw, nh = draw_bottle(img)
            labels.append(f"0 {cx:.5f} {cy:.5f} {nw:.5f} {nh:.5f}")

    # 4. Decide phones (1 phone, 65% chance)
    if random.random() < 0.65:
        cx, cy, nw, nh = draw_phone(img)
        labels.append(f"2 {cx:.5f} {cy:.5f} {nw:.5f} {nh:.5f}")

    # Apply lighting synthesis
    img = apply_lighting_transform(img, light_mode)

    # Save outputs
    img_dir = TRAIN_IMG_DIR if split == "train" else VAL_IMG_DIR
    lbl_dir = TRAIN_LBL_DIR if split == "train" else VAL_LBL_DIR
    filename = f"multilight_{split}_{index:05d}"

    img.save(img_dir / f"{filename}.jpg", quality=90)
    with open(lbl_dir / f"{filename}.txt", "w") as f:
        f.write("\n".join(labels))


def write_dataset_yaml():
    yaml_content = f"""# Multi-Lighting Can, Bottle, Phone & Human Dataset
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

nc: 4
names:
  0: bottle
  1: can
  2: phone
  3: person
"""
    with open(DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"[DATASET] Wrote data.yaml to {DATASET_DIR / 'data.yaml'}")


def main():
    print("=" * 65)
    print("  Generating Multi-Lighting Dataset (Bottles, Cans, Phones, Humans)")
    print("=" * 65)
    ensure_dirs()

    total_train = 800
    total_val = 200

    print(f"Generating {total_train} training scenes across 5 lighting modes...")
    for i in range(total_train):
        generate_single_sample(i, "train")
        if (i + 1) % 100 == 0:
            print(f"  Train: {i + 1}/{total_train} scenes generated...")

    print(f"Generating {total_val} validation scenes...")
    for i in range(total_val):
        generate_single_sample(i, "val")
        if (i + 1) % 50 == 0:
            print(f"  Val: {i + 1}/{total_val} scenes generated...")

    write_dataset_yaml()
    print("=" * 65)
    print(f"SUCCESS: 1,000 multi-lighting annotated scenes ready at:")
    print(f"  {DATASET_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
