import cv2
import numpy as np

# -----------------------
# Utilities
# -----------------------
def rgb_to_lab(img):
    """expects uint8 RGB image, returns float64 LAB (same scale as skimage rgb2lab)"""
    # OpenCV uses BGR; input assumed RGB -> convert to BGR for cv2 then to LAB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV L in [0,255], a and b in [0,255]. Convert to approx centered lab:
    # we will use simple Euclidean in this space; relative distances still meaningful.
    return lab

def rect_angle_to_yaw(rect):
    """Normalize cv2.minAreaRect angle to yaw in degrees ([-180,180))."""
    ((cx, cy), (w, h), angle) = rect
    # cv2 angle: angle of the box width relative to horizontal.
    # Convert to a consistent 'long-axis' angle:
    if w < h:
        yaw = angle + 90.0
    else:
        yaw = angle
    # normalize
    yaw = ((yaw + 180) % 360) - 180
    return yaw

# -----------------------
# Main detector
# -----------------------
def detect_belt_object_pose(img_rgb,
                            sample_box_size=20,
                            lab_threshold=18.0,
                            morph_kernel=5,
                            min_area_pixels=200):
    """
    img_rgb: uint8 RGB image (H,W,3)
    Returns: dict with keys:
      - 'center_px': (x,y) pixel center
      - 'w_px','h_px': rectangle sizes in pixels
      - 'angle_deg': yaw (deg) relative to image x axis
      - 'mask': uint8 mask used
      - 'contour': main contour (if any) else None
    """
    H, W, _ = img_rgb.shape
    lab = rgb_to_lab(img_rgb)  # (H,W,3) float32

    # Sample three small regions: top-middle, center-middle, bottom-middle
    cx = W // 2
    ys = [H // 8, H // 2, int(7 * H / 8)]
    samples = []
    half = sample_box_size // 2
    for y in ys:
        x0, x1 = cx - half, cx + half
        y0, y1 = y - half, y + half
        # clip
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        patch = lab[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        samples.append(np.mean(patch.reshape(-1, 3), axis=0))
    if len(samples) == 0:
        return None

    belt_color_lab = np.mean(np.vstack(samples), axis=0)

    # Color distance in LAB (use Euclidean in this opencv LAB-like space)
    diff = lab - belt_color_lab[None, None, :]
    dist = np.linalg.norm(diff, axis=2)

    # threshold => mask
    mask = (dist < lab_threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours, pick largest by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'mask': mask, 'contour': None}

    # pick largest contour by area
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < min_area_pixels:
        return {'mask': mask, 'contour': None, 'area': area}

    rect = cv2.minAreaRect(main)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    (cx_px, cy_px) = rect[0]
    (w_px, h_px) = rect[1]
    yaw_deg = rect_angle_to_yaw(rect)

    return {
        'center_px': (float(cx_px), float(cy_px)),
        'w_px': float(w_px),
        'h_px': float(h_px),
        'angle_deg': float(yaw_deg),
        'mask': mask,
        'contour': main,
        'box': box,
        'area': float(area),
        'belt_color_lab': belt_color_lab
    }

# -----------------------
# Conversion helpers
# -----------------------
def pixel_to_world_orthographic(center_px, image_shape, world_origin, meters_per_pixel):
    """
    For bird-eye orthographic camera:
      center_px: (x_px, y_px)
      image_shape: (H,W)
      world_origin: (x_world, y_world) corresponding to pixel origin (0,0) -- set to your map origin
      meters_per_pixel: scale
    Returns (x_world, y_world) in meters.
    """
    W = image_shape[1]
    H = image_shape[0]
    px, py = center_px
    # If origin is top-left, y increases downward: convert to a typical XY where +Y forward
    x_world = world_origin[0] + (px * meters_per_pixel)
    y_world = world_origin[1] + ((H - py) * meters_per_pixel)  # flip y if you want origin at bottom-left
    return (x_world, y_world)

def pixel_to_world_perspective(u, v, depth_image, intrinsics):
    """
    Convert pixel + depth to camera frame 3D point (x,y,z) in meters.
    intrinsics: dict with fx, fy, cx, cy
    depth_image: single-channel float depth (meters) same size as image
    """
    z = float(depth_image[int(round(v)), int(round(u))])
    if z <= 0:
        return None
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)
