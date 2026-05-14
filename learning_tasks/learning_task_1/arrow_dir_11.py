import cv2
import numpy as np

INPUT_IMAGE      = "arrow_right_2.jpg"

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
    
    # cv2.imshow("Debug 1: Original Image", binary)
    # cv2.waitKey(0)

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

# =========================================================================
# --- CHANGED SECTION 1: BULLETPROOF TEMPLATE MATCHING ---
# =========================================================================
def evaluate_half_with_template(patch_bin, side):
    h, w = patch_bin.shape
    half_w = w // 2
    
    # 1. Extract the specific half of the binary patch FIRST
    strip = patch_bin[:, :half_w] if side == 'left' else patch_bin[:, half_w:]
    
    # 2. Get the EXACT dimensions of whatever strip we just extracted 
    # (This completely prevents the bitwise size mismatch crash)
    strip_h, strip_w = strip.shape
    
    # 3. CREATE THE TEMPLATE: A solid white rectangle matching the exact strip size
    template = np.full((strip_h, strip_w), 255, dtype=np.uint8)
    
    # 4. Calculate Aspect Ratio of the template rectangle (Width / Height)
    template_aspect = float(strip_w) / strip_h if strip_h > 0 else 0
    
    # 5. FIT/SCORE THE TEMPLATE
    matching_pixels = cv2.bitwise_and(strip, template)
    score = np.sum(matching_pixels == 255) / (strip_h * strip_w) if (strip_h * strip_w) > 0 else 0
    
    return score, template_aspect


def detect_direction_from_contours(candidates):
    if not candidates:
        return None, 0, []
    sorted_by_area = sorted(candidates, key=lambda t: t[1], reverse=True)
    top2 = sorted_by_area[:2]
    top2.sort(key=lambda t: t[2])

    votes = {"left": 0, "right": 0}
    details = []

    for i, (c, area, x, y, w, h, br) in enumerate(top2):
        patch_bin = cv2.threshold(
            cv2.GaussianBlur(_big_gray[y:y+h, x:x+w], (3,3), 0),
            DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]

        left_score, left_aspect = evaluate_half_with_template(patch_bin, 'left')
        right_score, right_aspect = evaluate_half_with_template(patch_bin, 'right')
        
        print(f"\n--- Template Analysis for Shape {i+1} ---")
        print(f"Template Aspect Ratio: {left_aspect:.2f}")
        print(f"Left Half Rectangle Fit Score:  {left_score*100:.1f}%")
        print(f"Right Half Rectangle Fit Score: {right_score*100:.1f}%")

        if left_score < right_score:
            votes["left"] += 1
            print("Result: Triangle detected on the LEFT.")
        else:
            votes["right"] += 1
            print("Result: Triangle detected on the RIGHT.")

        details.append({
            "contour": c, "area": area, "x": x, "y": y, "w": w, "h": h,
            "patch_bin": patch_bin,
            "left_score": left_score, "right_score": right_score
        })

    if votes["right"] > votes["left"]:
        direction, confidence = "right", 1.0
    elif votes["left"] > votes["right"]:
        direction, confidence = "left", 1.0
    else:
        if len(details) >= 2:
            if details[1]["right_score"] < details[0]["left_score"]:
                direction, confidence = "right", 0.6
            else:
                direction, confidence = "left", 0.6
        else:
            d = details[0]
            if d["right_score"] < d["left_score"]:
                direction, confidence = "right", 0.5
            else:
                direction, confidence = "left", 0.5
    return direction, confidence, details

_big_gray = None

def detect_arrow(image_path):
    global _big_gray
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    original = preprocess_with_blur(original)
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    oh, ow = original.shape[:2]

    roi_color, (roi_x0, roi_y0) = get_roi(original)
    roi_gray,  _                = get_roi(orig_gray)

    big_color = upscale(roi_color)
    _big_gray = upscale(roi_gray)

    candidates, binary = find_arrow_contours(_big_gray)
    direction, confidence, details = detect_direction_from_contours(candidates)
    
    print("\n-------------------------------------")
    if direction:
        print(f"FINAL DECISION: Arrow pointing {direction.upper()} (Confidence: {confidence:.2f})")
    else:
        print("Could not determine arrow direction.")
    print("-------------------------------------\n")
        
    result_vis = original.copy()
    for d in details:
        ox = roi_x0 + d["x"] // SCALE
        oy = roi_y0 + d["y"] // SCALE
        cv2.rectangle(result_vis, (ox, oy),
                      (ox + d["w"]//SCALE, oy + d["h"]//SCALE), (0, 220, 0), 2)
        cv2.drawContours(result_vis, [d["contour"]], -1, (255, 0, 0), 2)

    if direction:
        cv2.putText(result_vis, f"ARROW: {direction.upper()}",
                    (roi_x0, roi_y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 0), 2)

    cv2.imshow("Arrow Detection Result", result_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return direction, confidence

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_IMAGE
    detect_arrow(path)