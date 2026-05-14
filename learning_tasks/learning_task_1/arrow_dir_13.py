
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROI_X_START      = 0.30
ROI_X_END        = 0.80
ROI_Y_START      = 0.40 
ROI_Y_END        = 0.70

SCALE            = 4
DARK_THRESHOLD   = 80
BLACK_RATIO_MIN  = 0.70
MIN_CONTOUR_AREA = 200
ASPECT_MIN       = 0.8
ASPECT_MAX       = 4.0

def preprocess_with_blur(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1,1), 0)  
    _, binary = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned = original.copy()
    for c in cnts:
        if cv2.contourArea(c) > 1000:  
            cv2.drawContours(cleaned, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    return cleaned

def get_roi(image):
    h, w = image.shape[:2]
    x0, x1 = int(w * ROI_X_START), int(w * ROI_X_END)
    y0, y1 = int(h * ROI_Y_START), int(h * ROI_Y_END)
    return image[y0:y1, x0:x1], (x0, y0)

def upscale(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

def find_arrow_contours(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        pix = gray[mask == 255]
        if len(pix) == 0:
            continue
        black_ratio = np.sum(pix <= DARK_THRESHOLD) / len(pix)
        if black_ratio < BLACK_RATIO_MIN:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        candidates.append((c, cv2.contourArea(c), x, y, w, h, black_ratio))

    if len(candidates) > 1:
        candidates.sort(key=lambda t: t[1], reverse=True) 
        candidates = [candidates[0]]                       

    return candidates, binary

# --- LOGIC 1: Y-SPAN DIFFERENCE ---
def detect_direction_from_contours(candidates, is_fallback=False):
    if not candidates:
        return None, 0, []
    
    # Just take the best candidate
    candidates.sort(key=lambda t: t[1], reverse=True)
    best_candidate = candidates[0]
    c, area, x, y, w, h, br = best_candidate

    mid_x = x + w // 2

    # Split contour points into left and right arrays based on X coordinate
    left_pts = c[c[:, 0, 0] < mid_x]
    right_pts = c[c[:, 0, 0] >= mid_x]

    # Calculate difference between highest and lowest Y pixel in each half
    left_diff = (np.max(left_pts[:, 0, 1]) - np.min(left_pts[:, 0, 1])) if len(left_pts) > 0 else 0
    right_diff = (np.max(right_pts[:, 0, 1]) - np.min(right_pts[:, 0, 1])) if len(right_pts) > 0 else 0

    print(f"Left Y-Span: {left_diff}px | Right Y-Span: {right_diff}px")

    if left_diff > right_diff:
        direction, confidence = "left", 1.0
    else:
        direction, confidence = "right", 1.0

    details = [{
        "contour": c, "area": area, "x": x, "y": y, "w": w, "h": h,
        "is_fallback": is_fallback
    }]

    return direction, confidence, details

_big_gray = None

def detect_arrow(image_path):
    global _big_gray
    original = cv2.imread(image_path)
    if original is None:
        # Handle missing files gracefully to avoid crashing the Matplotlib loop
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "File Missing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return blank, "Error"
    
    # --- SAVE A PURE COPY BEFORE THE BLUR DESTROYS IT ---
    pure_original = original.copy()
    pure_gray = cv2.cvtColor(pure_original, cv2.COLOR_BGR2GRAY)
    
    original = preprocess_with_blur(original)
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    oh, ow = original.shape[:2]

    roi_color, (roi_x0, roi_y0) = get_roi(original)
    roi_gray,  _                = get_roi(orig_gray)

    big_color = upscale(roi_color)
    _big_gray = upscale(roi_gray)

    candidates, binary = find_arrow_contours(_big_gray)
    is_fallback = False

    # --- BOTTOM FALLBACK LOGIC ---
    if not candidates:
        print("No contours found in scaled ROI. Checking Bottom Fallback...")
        offset_y = int(oh * 0.6)
        
        # --- USE THE UN-MUTATED RAW IMAGE HERE ---
        bottom_half = pure_gray[offset_y:oh, 0:ow]
        
        # Inverted thresholding at 75 (darker than 75 becomes white/255)
        _, fb_binary = cv2.threshold(bottom_half, 75, 255, cv2.THRESH_BINARY_INV)
        fb_cnts, _ = cv2.findContours(fb_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if fb_cnts:
            largest_c = max(fb_cnts, key=cv2.contourArea)
            if cv2.contourArea(largest_c) > 50: # Basic noise filter
                
                # Shift contour coordinates back down into the global image space
                largest_c = largest_c + np.array([[[0, offset_y]]])
                
                fx, fy, fw, fh = cv2.boundingRect(largest_c)
                candidates = [(largest_c, cv2.contourArea(largest_c), fx, fy, fw, fh, 1.0)]
                is_fallback = True

    direction, confidence, details = detect_direction_from_contours(candidates, is_fallback)
    
    if direction:
        print(f"Arrow direction: {direction.upper()} (confidence={confidence:.2f})")
    else:
        print("Could not determine arrow direction.")
        
    # --- DRAW ON THE PURE IMAGE, NOT THE MUTATED ONE ---
    result_vis = pure_original.copy()
    
    for d in details:
        if d["is_fallback"]:
            # Fallback coords are already 1:1 scale with the original image
            ox, oy, ow_disp, oh_disp = d["x"], d["y"], d["w"], d["h"]
            text_pos = (ox, max(oy - 10, 20))
        else:
            # Standard ROI coords need to be downscaled and offset
            ox = roi_x0 + d["x"] // SCALE
            oy = roi_y0 + d["y"] // SCALE
            ow_disp = d["w"] // SCALE
            oh_disp = d["h"] // SCALE
            text_pos = (roi_x0, roi_y0 - 10)

        cv2.rectangle(result_vis, (ox, oy), (ox + ow_disp, oy + oh_disp), (0, 220, 0), 2)
        
        # Drawing the raw contour directly onto result_vis
        cv2.drawContours(result_vis, [d["contour"]], -1, (255, 0, 0), 2)

    if direction:
        cv2.putText(result_vis, f"ARROW: {direction.upper()}",
                    text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 0), 2)

    # Return the processed image and the text string instead of showing it
    return result_vis, direction if direction else "Not Found"

# ==========================================
# MATPLOTLIB BATCH PROCESSING
# ==========================================
if __name__ == "__main__":
    print("Starting Batch Processing Pipeline...")

    image_files = [
        "arrow_left.jpg", "arrow_right.jpg", 
        "arrow_left_2.jpg", "arrow_right_2.jpg", 
        "arrow_left_3.jpg", "arrow_right_3.jpg"
    ]

    # Set up a 3x2 grid in Matplotlib
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Y-Span Arrow Detection Results')

    # Flatten the axes array to easily iterate over it
    axes = axes.flatten()

    for i, filename in enumerate(image_files):
        print(f"\nProcessing: {filename}...")
        
        # Pass the image through our pipeline
        processed_img, direction = detect_arrow(filename)
        
        # OpenCV uses BGR, but Matplotlib requires RGB format to display colors correctly
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Plot the image on the corresponding grid square
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"File: {filename}\nResult: {direction.upper()}", fontweight="bold")
        axes[i].axis('off')  # Hide the X and Y axis tick marks

    # Adjust layout so titles don't overlap and show the final plot
    plt.tight_layout()
    print("\nRendering Grid...")
    plt.show()