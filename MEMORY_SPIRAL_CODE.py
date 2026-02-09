import math
import os
from PIL import Image, ImageDraw, ImageFont

# For file selection dialog
import tkinter as tk
from tkinter import filedialog

# ---------------------------
# CONFIGURATION
# ---------------------------

# Output & animation
CANVAS_SIZE = 800          # square canvas: 800x800 pixels
MARGIN = 40                # margin around spiral within canvas (px)

FPS = 30                   # frames per second
DURATION_S = 5.0           # total animation length in seconds
NUM_FRAMES = int(FPS * DURATION_S)  # 150 frames
FRAME_DURATION_MS = int(1000 / FPS)  # ms per frame

# Spiral geometry (in abstract 'world' units before scaling to canvas)
INITIAL_RADIUS = 0.0       # start at center
COIL_SPACING = 3.0         # radial distance between successive coils
CHAR_SPACING = 1.0         # arc-length distance between characters

# Text appearance
FONT_SIZE = 18             # base font size in pixels (apparent size changes with zoom)
FONT_PATH = None           # or a .ttf path, e.g. "Helvetica.ttf"
TEXT_COLOR = (0, 0, 0, 255)
ANOMALY_COLOR = (255, 0, 0, 255)   # red for anomaly chars
BG_COLOR = (255, 255, 255)

# Temporal anomaly (slowdown) parameters
SLOW_CHARS_NOMINAL = 150    # ~100 chars/s for 1.5s
SLOW_DURATION = 1.5         # duration of slowdown (seconds)

# Geometric anomaly (loosening) parameters:
# One-sided bulge that expands outward and never returns inward
BULGE_AMP = 0.4             # maximum fractional radius expansion
BULGE_SIGMA = 0.04          # controls how quickly the expansion ramps up

# Coloring anomaly width
ANOMALY_COLOR_WIDTH = BULGE_SIGMA * 2.2


# ---------------------------
# Helpers
# ---------------------------

def load_font(size):
    """Load a TrueType font if provided, else default PIL font."""
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except OSError:
            print("Warning: could not load custom font, using default PIL font.")
    return ImageFont.load_default()


def generate_spiral_positions(num_chars,
                              char_spacing=CHAR_SPACING,
                              initial_radius=INITIAL_RADIUS,
                              coil_spacing=COIL_SPACING):
    """
    Generate positions along an Archimedean spiral for num_chars characters.
    """
    a = initial_radius
    b = coil_spacing / (2.0 * math.pi)

    points = []
    theta = 0.0
    r = a + b * theta
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    last_x, last_y = x, y
    accumulated_length = 0.0
    target_length = 0.0
    step_theta = 0.02

    while len(points) < num_chars:
        theta += step_theta
        r = a + b * theta
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        dx = x - last_x
        dy = y - last_y
        ds = math.hypot(dx, dy)

        accumulated_length += ds

        while accumulated_length >= target_length and len(points) < num_chars:
            excess = accumulated_length - target_length
            t = 1.0 - excess / ds if ds != 0 else 0.0
            px = last_x + t * dx
            py = last_y + t * dy
            points.append((px, py, theta))
            target_length += char_spacing

        last_x, last_y = x, y

    return points


def compute_tangent_angles(points):
    """Compute rotation angle of each character along the spiral."""
    n = len(points)
    angles_deg = []
    for i in range(n):
        if i == 0:
            x1, y1, _ = points[i]
            x2, y2, _ = points[i + 1]
        elif i == n - 1:
            x1, y1, _ = points[i - 1]
            x2, y2, _ = points[i]
        else:
            x1, y1, _ = points[i - 1]
            x2, y2, _ = points[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dy, dx)
        angles_deg.append(math.degrees(angle_rad))
    return angles_deg


# ---------------------------
# Temporal mapping (slowdown)
# ---------------------------

def chars_revealed_at_time(t, num_chars, anomaly_frac,
                           slow_chars_nominal=SLOW_CHARS_NOMINAL,
                           slow_duration=SLOW_DURATION):
    """Piecewise-linear reveal based on anomaly-centered slowdown."""
    D = DURATION_S
    N = num_chars
    if N <= 0:
        return 0.0
    if N == 1:
        return 1.0 if t >= 0 else 0.0

    a = max(0.0, min(1.0, anomaly_frac))

    # Text space window
    i0 = a * (N - 1)
    W_chars = min(float(slow_chars_nominal), float(N))
    halfW = W_chars / 2
    i1 = max(0.0, i0 - halfW)
    i2 = min(float(N), i0 + halfW)
    W_eff = max(1.0, i2 - i1)

    # Time window
    t_center = a * D
    half_T = slow_duration / 2.0
    t1 = max(0.0, t_center - half_T)
    t2 = min(D,   t_center + half_T)

    if t2 <= t1:
        return N * min(1.0, max(0.0, t / D))

    slow_slope = W_eff / (t2 - t1)
    pre_slope  = i1 / t1 if t1 > 0 else 0.0
    post_slope = (N - i2) / (D - t2) if D > t2 else 0.0

    if t <= 0:
        C = 0.0
    elif t < t1:
        C = pre_slope * t
    elif t <= t2:
        C = i1 + slow_slope * (t - t1)
    elif t < D:
        C = i2 + post_slope * (t - t2)
    else:
        C = float(N)

    return max(0.0, min(float(N), C))


# ---------------------------
# One-sided bulge (no overlap)
# ---------------------------

def compute_bulge_deltas(num_chars, anomaly_frac):
    """
    Produce a monotonic bulge that:
    - stays 0 before anomaly
    - increases smoothly after anomaly
    - never decreases (so no overlap)
    """
    if num_chars <= 0:
        return []
    if num_chars == 1:
        return [0.0]

    a = max(0.0, min(1.0, anomaly_frac))
    deltas_raw = [0.0] * num_chars

    for i in range(num_chars):
        t_char = i / (num_chars - 1)
        if t_char <= a:
            delta = 0.0
        else:
            d = (t_char - a) / max(1e-9, BULGE_SIGMA)
            delta = BULGE_AMP * (1.0 - math.exp(-0.5 * d * d))
        deltas_raw[i] = delta

    # enforce monotonic increasing
    deltas = []
    max_so_far = 0.0
    for d in deltas_raw:
        if d > max_so_far:
            max_so_far = d
        deltas.append(max_so_far)
    return deltas


# ---------------------------
# Main animation logic
# ---------------------------

def create_spiral_gif(text, anomaly_pct, output_path="spiral_growing.gif"):
    """Render the animated spiral."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    chars = list(text)
    num_chars = len(chars)

    if num_chars == 0:
        print("No text entered.")
        return

    anomaly_frac = max(0.0, min(1.0, anomaly_pct / 100.0))
    print(f"Anomaly at: {anomaly_pct}%")

    positions = generate_spiral_positions(num_chars)
    tangent_angles = compute_tangent_angles(positions)
    base_radii = [math.hypot(x, y) for (x, y, _) in positions]

    bulge_deltas = compute_bulge_deltas(num_chars, anomaly_frac)

    font = load_font(FONT_SIZE)
    canvas_center = CANVAS_SIZE // 2
    max_r_canvas = canvas_center - MARGIN

    frames = []

    for frame_idx in range(NUM_FRAMES):
        t = frame_idx / FPS
        C = chars_revealed_at_time(t, num_chars, anomaly_frac)
        max_char_index = max(1, min(num_chars, int(C)))

        # Determine zoom
        visible_adj_r = []
        for i in range(max_char_index):
            r0 = base_radii[i] or 1e-6
            r_adj = r0 * (1 + bulge_deltas[i])
            visible_adj_r.append(r_adj)

        current_max_r = max(visible_adj_r) if visible_adj_r else 1.0
        scale = max_r_canvas / current_max_r if current_max_r else 1.0

        frame = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)

        for i in range(max_char_index):
            ch = chars[i]
            if ch == " ":
                continue

            x0, y0, _ = positions[i]
            r0 = base_radii[i] or 1e-6

            t_char = i / (num_chars - 1) if num_chars > 1 else 0.0

            # Apply bulge
            delta = bulge_deltas[i]
            x = x0 * (1 + delta)
            y = y0 * (1 + delta)

            # Map to canvas
            sx = canvas_center + int(x * scale)
            sy = canvas_center - int(y * scale)

            # Choose color
            if abs(t_char - anomaly_frac) <= ANOMALY_COLOR_WIDTH:
                color = ANOMALY_COLOR
            else:
                color = TEXT_COLOR

            angle_deg = tangent_angles[i]

            # Render character patch
            dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
            ddraw = ImageDraw.Draw(dummy)
            bbox = ddraw.textbbox((0, 0), ch, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            pad = 4
            glyph = Image.new("RGBA", (w + 2*pad, h + 2*pad), (0, 0, 0, 0))
            gdraw = ImageDraw.Draw(glyph)
            gdraw.text((pad, pad), ch, font=font, fill=color)

            rotated = glyph.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
            rw, rh = rotated.size

            frame.paste(rotated, (int(sx - rw/2), int(sy - rh/2)), rotated)

        frames.append(frame.convert("P", palette=Image.ADAPTIVE))

        print(f"Frame {frame_idx+1}/{NUM_FRAMES} — chars shown: {max_char_index}/{num_chars}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0
    )

    print(f"\n✨ Saved spiral animation as {os.path.abspath(output_path)} ✨")


# ---------------------------
# File selection instead of terminal input
# ---------------------------

def get_text_from_file_dialog():
    """
    Open a file dialog to select a .txt file and return its contents as a string.
    """
    root = tk.Tk()
    root.withdraw()  # hide main window

    file_path = filedialog.askopenfilename(
        title="Select your LLM text file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    root.destroy()

    if not file_path:
        print("No file selected.")
        return ""

    print(f"Selected file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # fallback if encoding is different
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    paragraph = get_text_from_file_dialog()
    if not paragraph.strip():
        print("No text loaded from file. Exiting.")
    else:
        raw = input("Anomaly percentage (0–100): ").strip()
        try:
            anomaly_pct = float(raw)
        except:
            anomaly_pct = 70.0

        anomaly_pct = max(0, min(100, anomaly_pct))
        create_spiral_gif(paragraph, anomaly_pct, "spiral_growing.gif")
