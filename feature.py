import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv, hsv2rgb
from skimage.exposure import equalize_adapthist

# ======================================================================
# ----------------------------- CORE FUNCTIONS -------------------------
# ======================================================================

def boost_brightness_saturation(img_in, satBoost, valBoost):
    """Enhances saturation and brightness of an RGB image."""
    is_uint8 = img_in.dtype == np.uint8
    img_d = img_in.astype(float) / 255.0 if is_uint8 else img_in.copy()
    hsv = rgb2hsv(img_d)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    S = np.clip(S * satBoost, 0, 1)
    V = np.clip(V * valBoost, 0, 1)
    hsv_boosted = np.stack([H, S, V], axis=-1)
    img_out = hsv2rgb(hsv_boosted)
    return (np.clip(img_out, 0, 1) * 255).astype(np.uint8) if is_uint8 else img_out


def autoLabBackgroundRemove(img):
    """
    Automatically removes background using Euclidean distance in LAB space.
    Returns: img_fg (masked image), mask_uint8 (the correct binary mask), th
    """
    img_d = img.astype(float) / 255.0
    lab = rgb2lab(img_d)
    H, W, _ = lab.shape
    corners = np.array([lab[0,0], lab[0,W-1], lab[H-1,0], lab[H-1,W-1]])
    bg = np.mean(corners, axis=0)
    th, thStep, thMax = 5.0, 2.0, 150.0
    while th < thMax:
        D = np.sqrt(np.sum((lab - bg)**2, axis=2))
        mask = D > th
        border = np.concatenate([mask[0], mask[-1], mask[:,0], mask[:,-1]])
        if np.all(border == False):
            break
        th += thStep
    mask_uint8 = (mask.astype(np.uint8) * 255)
    img_fg = (img_d * mask[..., None]) 
    return (img_fg * 255).astype(np.uint8), mask_uint8, th


def maskBoundaryPoints(mask, nPoints):
    """Subsamples boundary points evenly along the perimeter."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((nPoints, 2))

    pts_full = max(contours, key=len).squeeze() 
    if pts_full.ndim == 1:
        pts_full = pts_full.reshape(-1, 2)

    diffs = np.diff(pts_full, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumDist = np.concatenate([[0], np.cumsum(dists)])

    L = cumDist[-1]
    target = np.linspace(0, L, nPoints, endpoint=False)

    pts_x = np.interp(target, cumDist, pts_full[:, 0])
    pts_y = np.interp(target, cumDist, pts_full[:, 1])

    return np.vstack([pts_x, pts_y]).T


def shapeDescriptors(pts, nFourier=10):
    """
    Computes PCA and Fourier descriptors for the boundary shape (pts).
    Returns: (Fdesc_normalized, eigvals_normalized, eig_ratio)
    """
    pts_c = pts - np.mean(pts, axis=0)
    C = np.cov(pts_c, rowvar=False)
    eigvals, _ = np.linalg.eigh(C)
    eigvals = np.sort(eigvals)[::-1]
    
    # PCA Features (calculated on smooth boundary points 'pts')
    eig_ratio = eigvals[0] / eigvals[1] if eigvals[1] > 0 else 0.0
    eigvals_norm = eigvals / np.sum(eigvals)
    
    # Fourier Features (Absolute, Normalized)
    z = pts_c[:, 0] + 1j * pts_c[:, 1]
    Z = np.fft.fft(z)
    
    # We take N modes starting from the first non-DC term (Z[1])
    Fdesc_abs = np.abs(Z[1:nFourier+1])
    Fdesc_norm = Fdesc_abs / np.sum(Fdesc_abs)
    
    return Fdesc_norm, eigvals_norm, eig_ratio


def generate_combined_canny_edges(img):
    """Generates the combined Canny edges without initial masking."""
    img_float = img.astype(np.float32) / 255
    R = equalize_adapthist(img_float[:,:,0])
    B = equalize_adapthist(img_float[:,:,2])
    eR_temp = cv2.Laplacian(R, cv2.CV_32F)
    eB_temp = cv2.Laplacian(B, cv2.CV_32F)
    edgeColor_R = np.median(R[np.abs(eR_temp) > 0])
    edgeColor_B = np.median(B[np.abs(eB_temp) > 0])
    tol = 0.1
    R_adj = np.abs(R - edgeColor_R) > tol
    B_adj = np.abs(B - edgeColor_B) > tol
    edges_R = cv2.Canny((R_adj*255).astype(np.uint8), 100, 200)
    edges_B = cv2.Canny((B_adj*255).astype(np.uint8), 100, 200)
    edges_combined = edges_R | edges_B
    return edges_combined

def order_rectangle_points(pts):
    """Orders 4 rectangle points clockwise."""
    pts = np.array(pts).reshape(4, 2)
    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    order = np.argsort(angles)
    return pts[order]

# ======================================================================
# ------------------------- ASPECT RATIO & COLOR -----------------------
# ======================================================================

def calculate_aspect_ratio(shape_type, points):
    """Calculates the aspect ratio (Height / Width) for the global best fit shape."""
    
    # --- Aspect Ratio for RECTANGLE ---
    if shape_type == "Rectangle":
        rect = cv2.minAreaRect(points)
        width, height = rect[1]
        if width == 0: return 0.0
        return height / width 

    # --- Aspect Ratio for TRIANGLE ---
    elif shape_type == "Triangle":
        if len(points.shape) == 3: points = points.squeeze()
        if points.size < 6: return 0.0
        
        x, y, w, h = cv2.boundingRect(points)
        
        if w == 0: return 0.0
        return h / w
        
    # --- Aspect Ratio for CIRCLE ---
    elif shape_type == "Circle":
        return 1.0
        
    return 0.0


def calculate_color_features(img_rgb, boundary_mask):
    """Computes mean L, a, and b values within the object boundary."""
    img_lab = rgb2lab(img_rgb.astype(float) / 255.0)
    
    L_flat = img_lab[:, :, 0][boundary_mask > 0]
    a_flat = img_lab[:, :, 1][boundary_mask > 0]
    b_flat = img_lab[:, :, 2][boundary_mask > 0]
    
    if L_flat.size == 0:
        return np.array([0.0, 0.0, 0.0])

    mean_L = np.mean(L_flat)
    mean_a = np.mean(a_flat)
    mean_b = np.mean(b_flat)
    
    return np.array([mean_L, mean_a, mean_b])


# ======================================================================
# ------------------------- LOCAL SHAPE ANALYSIS -----------------------
# ======================================================================

def fit_local_shapes_pca(points):
    """
    Splits the point cloud along the max PCA direction and fits the 
    minimum area bounding shape (Rect, Tri, or Circle) to each half.
    
    Returns: list of tuples: [(shape_type, shape_coordinates, area, half_points), ...]
    """
    if points.shape[0] < 3:
        return []

    # 1. PCA: Find max PCA direction (using Canny points for split)
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    max_eigenvector = eigvecs[:, -1] 
    
    # 2. Split the point cloud
    points_centered = points - mean
    projections = points_centered.dot(max_eigenvector)
    
    half1_points = points[projections <= 0]
    half2_points = points[projections > 0]
    
    local_shapes_data = []

    # 3. Fit and select minimum area shape for each half
    for half_points in [half1_points, half2_points]:
        if half_points.shape[0] >= 3:
            
            # --- Shape Fitting (as before) ---
            rect = cv2.minAreaRect(half_points)
            rect_pts = cv2.boxPoints(rect).astype(np.int32)
            rect_area = cv2.contourArea(rect_pts)

            ret_tri, tri_pts = cv2.minEnclosingTriangle(half_points)
            tri_pts_int = np.int32(tri_pts)
            tri_area = ret_tri

            (cc, rc) = cv2.minEnclosingCircle(half_points)
            circle_area = np.pi * rc * rc
            
            # --- Selection (Minimum Area) ---
            areas = np.array([rect_area, tri_area, circle_area])
            min_index = np.argmin(areas)
            min_area = areas[min_index]
            
            if min_index == 0:
                local_shapes_data.append(("Rectangle", rect_pts, min_area, half_points))
            elif min_index == 1:
                local_shapes_data.append(("Triangle", tri_pts_int, min_area, half_points))
            elif min_index == 2:
                cc_int = (int(cc[0]), int(cc[1]))
                local_shapes_data.append(("Circle", (cc_int, int(rc)), min_area, half_points))
            
    return local_shapes_data


# ======================================================================
# ----------------------------- MAIN PIPELINE ---------------------------
# ======================================================================

def generate_pipeline_features():

    # --- Setup and Image Loading (Assumes 'img2.jpeg' exists in the running directory) ---
    try:
        img_bgr = cv2.imread("img3.jpg") 
        if img_bgr is None:
            raise FileNotFoundError()
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        # Dummy image for testing purposes if run directly
        print("Image 'img2.jpeg' not found. Returning empty features.")
        return np.zeros(22)
    
    img_resized = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
    img_rgb = img_resized.copy()
    H, W, _ = img_rgb.shape 

    # --- Feature Extraction Steps ---
    img_boosted = boost_brightness_saturation(img_rgb, 3, 3)
    img_fg, mask_uint8, th = autoLabBackgroundRemove(img_boosted)
    pts = maskBoundaryPoints(mask_uint8, 50)

    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    smooth_mask_int = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(smooth_mask_int, [pts_int], 255)

    edges_combined = generate_combined_canny_edges(img_fg)
    edges_masked = edges_combined.astype(bool) & (smooth_mask_int > 0)
    edges_masked_uint8 = (edges_masked.astype(np.uint8) * 255)

    canny_y, canny_x = np.where(edges_masked_uint8 > 0)
    boundary_points_int = np.column_stack((canny_x, canny_y)).astype(np.int32)
    
    # [1-3] PCA/Fourier Features
    nFourierModes = 10
    Fdesc_norm, eigvals_norm, eig_ratio = shapeDescriptors(pts, nFourier=nFourierModes)
    
    # Initialize Global/Local Features
    global_shape_num = 0.0
    global_aspect_ratio = 0.0
    #print(np.shape(boundary_points_int))
    if boundary_points_int.size >= 3:
        # Global Fitting
        hull = cv2.convexHull(boundary_points_int)
        hull_area = cv2.contourArea(hull)
        
        rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(boundary_points_int)).astype(np.int32))
        ret, tri_pts = cv2.minEnclosingTriangle(boundary_points_int)
        tri_area = ret
        circle_area = np.pi * cv2.minEnclosingCircle(boundary_points_int)[1]**2
        
        areas = np.array([rect_area, tri_area, circle_area])
        ratios = hull_area / areas
        best = np.argmin(np.abs(ratios - 1))
        
        best_name = ["Rectangle", "Triangle", "Circle"][best]
        # Mapping: Rectangle=3.0, Triangle=2.0, Circle=1.0
        global_shape_num = float([3.0, 2.0, 1.0][best])

        # Calculate Aspect Ratio (using Height/Width)
        global_aspect_ratio = calculate_aspect_ratio(best_name, boundary_points_int)
            
        # Local Fitting
        local_shapes_data = fit_local_shapes_pca(boundary_points_int)
    
    # Color Features [20-22]
    mean_LAB = calculate_color_features(img_rgb, smooth_mask_int)

    # --- FINAL FEATURE VECTOR ASSEMBLY ---
    
    local_features_req = np.zeros(4) # [Local1 ID, Local1 Ratio, Local2 ID, Local2 Ratio]
    
    if len(local_shapes_data) == 2:
        local_shape_map = {"Circle": 1.0, "Triangle": 2.0, "Rectangle": 3.0}
        
        for i, (shape_type, coords, area, half_points) in enumerate(local_shapes_data):
            shape_num = local_shape_map[shape_type]
            
            # --- UPDATED LOCAL RATIO CALCULATION: Local Hull Area / Local Fitted Area ---
            ratio = 0.0
            if half_points.shape[0] >= 3 and area > 0:
                local_hull = cv2.convexHull(half_points)
                local_hull_area = cv2.contourArea(local_hull)
                ratio = local_hull_area / area
            # -----------------------------------
            
            local_features_req[i * 2] = shape_num
            local_features_req[i * 2 + 1] = ratio

    
    FINAL_FEATURE_VECTOR = np.concatenate([
        eigvals_norm[:2],                      # 1, 2: Normalized Eigenvalues (v1, v2)
        [eig_ratio],                           # 3: Eigen Ratio
        Fdesc_norm,                            # 4-13: Fourier Modes (10)
        [global_shape_num],                    # 14: Global Shape ID (1/2/3)
        [global_aspect_ratio],                 # 15: Global Aspect Ratio
        local_features_req,                    # 16-19: Local Shape ID 1, Ratio 1, ID 2, Ratio 2
        mean_LAB                               # 20-22: Mean L, a, b
    ])
    
    
    # ---------------------- 2×4 Visualization Grid ----------------------
    fig = plt.figure(figsize=(22, 10))

    # ============================================================
    # 1. ORIGINAL + RED BOUNDARY POINTS
    # ============================================================
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.set_title("1. Original (Boundary Marked)")
    ax1.imshow(img_rgb)
    ax1.axis('off')
    # Boundary scatter — red with thickness
    # ============================================================
    # 2. BOOSTED IMAGE
    # ============================================================
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.set_title("2. Boosted")
    ax2.imshow(img_boosted)
    ax2.axis('off')

    # ============================================================
    # 3. FOREGROUND + RED BOUNDARY POINTS
    # ============================================================
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.set_title("3. Foreground + Boundary Points")
    ax3.imshow(img_fg)

    # Boundary scatter — red with thickness
    if pts.shape[0] > 0:
        ax3.scatter(pts[:,0], pts[:,1], c='red', s=14)

    ax3.axis('off')

    # ============================================================
    # 4. CANNY WHITE POINTS + LIME BOUNDARY LOOP ON BLACK
    # ============================================================
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.set_title("4. Canny Points + Boundary Loop")

    # Black background
    ax4.imshow(np.zeros_like(edges_combined), cmap="gray")

    # Canny points = thin white dots
    if boundary_points_int.size > 0:
        ax4.scatter(boundary_points_int[:,0], boundary_points_int[:,1],
                    c='white', s=1,linewidth=0.01)

    # Lime boundary loop from maskBoundaryPoints()
    if pts.shape[0] > 0:
        ax4.plot(
            np.r_[pts[:,0], pts[0,0]],
            np.r_[pts[:,1], pts[0,1]],
            color='limegreen', linewidth=1.3
    )
    #ax4.plot(pts[:,0], pts[:,1], color='red', linewidth=1.5)

    ax4.axis('off')

    # ============================================================
    # 5. GLOBAL CONVEX HULL (GREEN)
    # ============================================================
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.set_title("5. Global Convex Hull")
    ax5.imshow(img_rgb)

    if boundary_points_int.size > 0:
        hull_global = cv2.convexHull(boundary_points_int)
        ax5.plot(hull_global[:,0,0], hull_global[:,0,1],
                '-c', linewidth=1.3)

    ax5.axis('off')

    # ============================================================
    # SUPPORT FUNCTION: ORDER RECTANGLE POINTS CLOCKWISE
    # ============================================================
    def order_rectangle_points(pts):
        pts = np.array(pts).reshape(4, 2)
        cx, cy = np.mean(pts, axis=0)
        ang = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
        return pts[np.argsort(ang)]

    # ============================================================
    # 6. ALL GLOBAL SHAPE FITS (RECT → RED, TRIANGLE → ORANGE, CIRCLE → YELLOW)
    # ============================================================
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.set_title("6. All Global Fits")
    ax6.imshow(img_rgb)
    #print(np.shape(boundary_points_int))
    if boundary_points_int.size > 0:

        # Rectangle (RED)
        rect_pts = order_rectangle_points(cv2.boxPoints(
            cv2.minAreaRect(boundary_points_int)))
        ax6.plot(rect_pts[:,0], rect_pts[:,1], color='green', linewidth=1.3)
        ax6.plot(
            [rect_pts[i,0] for i in [0,1,2,3,0]],
            [rect_pts[i,1] for i in [0,1,2,3,0]],
            color='limegreen', linewidth=1.5
        )

        # Triangle (ORANGE)
        ret, tri_raw = cv2.minEnclosingTriangle(boundary_points_int)
        tri = tri_raw.squeeze().astype(int)
        ax6.plot(
            [tri[0,0], tri[1,0], tri[2,0], tri[0,0]],
            [tri[0,1], tri[1,1], tri[2,1], tri[0,1]],
            '-', color='red', linewidth=1.3
        )

        # Circle (YELLOW)
        (cx, cy), r = cv2.minEnclosingCircle(boundary_points_int)
        circ = plt.Circle((cx, cy), r, edgecolor='yellow',
                        facecolor='none', linewidth=1.3)
        ax6.add_patch(circ)

    ax6.axis('off')

    # ============================================================
    # 7. BEST GLOBAL FIT ONLY
    # ============================================================
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_title(f"7. Best Global Fit: {best_name}")
    ax7.imshow(img_rgb)

    if best_name == "Rectangle":
        ax7.plot(rect_pts[:,0], rect_pts[:,1], color='green', linewidth=1.3)

    elif best_name == "Triangle":
        ax7.plot(
            [tri[0,0], tri[1,0], tri[2,0], tri[0,0]],
            [tri[0,1], tri[1,1], tri[2,1], tri[0,1]],
            '-', color='red', linewidth=1.3
        )

    elif best_name == "Circle":
        circ2 = plt.Circle((cx, cy), r, edgecolor='yellow',
                        facecolor='none', linewidth=1.3)
        ax7.add_patch(circ2)

    ax7.axis('off')

    # ============================================================
    # 8. LOCAL FITS H1 & H2 (R/T/C)
    # ============================================================
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.set_title("8. Local Fits H1 & H2")
    ax8.imshow(img_rgb)

    for shape_type, coords, area, half_pts in local_shapes_data:

        # Rectangle
        if shape_type == "Rectangle":
            rect = order_rectangle_points(coords)
            ax8.plot(rect[:,0], rect[:,1], color='cyan', linewidth=1.5)
            ax8.plot(
                [rect[i,0] for i in [0,1,2,3,0]],
                [rect[i,1] for i in [0,1,2,3,0]],
                color='cyan', linewidth=1.3
            )

        # Triangle
        elif shape_type == "Triangle":
            tri = np.array(coords).reshape(-1,2)
            if tri.shape == (3,2):
                ax8.plot(
                    [tri[0,0], tri[1,0], tri[2,0], tri[0,0]],
                    [tri[0,1], tri[1,1], tri[2,1], tri[0,1]],
                    '-', color='red', linewidth=1.3
                )

        # Circle
        elif shape_type == "Circle":
            (ccx, ccy), rr = coords
            circ = plt.Circle((ccx, ccy), rr, edgecolor='lime',
                            facecolor='none', linewidth=1.5)
            ax8.add_patch(circ)

    ax8.axis('off')

    plt.tight_layout(pad=1.2)
    plt.show()
    return FINAL_FEATURE_VECTOR


if __name__ == "__main__":
    # If run directly, it will return an empty vector unless 'img2.jpeg' is provided
    print(generate_pipeline_features())
    pass