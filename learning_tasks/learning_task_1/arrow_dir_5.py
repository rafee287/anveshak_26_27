import cv2
import numpy as np

img_orig = cv2.imread("arrow_left_3.jpg")

if img_orig is None:
    print("Error: Could not load image.")
else:
    # --- DEBUG: Original Image ---
    cv2.imshow("Debug 1: Original Image", img_orig)
    cv2.waitKey(0)

    height, width = img_orig.shape[:2]
    new_height = int(1.5 * height)
    new_width = int(1.5 * width)
    img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    gray_zoomed = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

    binary_zoomed = cv2.adaptiveThreshold(
        gray_zoomed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # --- DEBUG: Adaptive Threshold ---
    cv2.imshow("Debug 2: Adaptive Threshold Mask", binary_zoomed)
    cv2.waitKey(0)

    kernel = np.ones((1, 1), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_zoomed, cv2.MORPH_CLOSE, kernel)

    # --- DEBUG: Cleaned Mask ---
    cv2.imshow("Debug 3: Mask After Morphological Closing", cleaned_mask)
    cv2.waitKey(0)

    contours_zoomed, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    output_img_zoomed = img_zoomed.copy()

    for cnt in contours_zoomed:
        x, y, w, h = cv2.boundingRect(cnt)
        touches_edge = (x <= 1 or y <= 1 or (x + w) >= (new_width - 1) or (y + h) >= (new_height - 1))
        
        if not touches_edge:
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            
            if area > 50 and length < 1000: 
                hull_indices = cv2.convexHull(cnt, returnPoints=False)
                
                if len(hull_indices) > 3:
                    # --- THE FIX: The Try/Except Shield ---
                    try:
                        # Attempt to calculate defects
                        defects = cv2.convexityDefects(cnt, hull_indices)
                    except cv2.error:
                        # If the contour self-intersects and throws the "monotonous" error,
                        # it is just noisy garbage. Ignore it and skip to the next contour!
                        continue 
                    # --------------------------------------
                    
                    if defects is not None:
                        deep_valley_count = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            actual_depth = d / 256.0
                            minimum_depth_required = w * 0.08
                            
                            if actual_depth > minimum_depth_required:
                                deep_valley_count += 1
                                farthest_point = tuple(cnt[f][0])
                                cv2.circle(output_img_zoomed, farthest_point, 4, (255, 0, 0), -1)
                        
                        if deep_valley_count == 2:
                            valid_contours.append(cnt)

    cv2.drawContours(output_img_zoomed, valid_contours, -1, (0, 255, 0), 2)
    
    # --- DEBUG: Filtered Contours (Did the arrow survive the defect test?) ---
    cv2.imshow("Debug 4: Valid Contours & Armpits Found", output_img_zoomed)
    cv2.waitKey(0)

    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(largest_contour)
        box_center_x = int(rect[0][0])
        box_center_y = int(rect[0][1])
        
        isolated_arrow_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
        cv2.drawContours(isolated_arrow_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        dist_transform = cv2.distanceTransform(isolated_arrow_mask, cv2.DIST_L2, 5)
        
        # --- DEBUG: Distance Transform Heatmap ---
        # Note: We must normalize the heatmap to 0-1 so OpenCV can display the floats!
        cv2.imshow("Debug 5: Distance Transform Heatmap", cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX))
        cv2.waitKey(0)
        
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        cX_head, cY_head = max_loc[0], max_loc[1]

        cv2.circle(output_img_zoomed, (box_center_x, box_center_y), 4, (0, 255, 0), -1)
        cv2.circle(output_img_zoomed, (cX_head, cY_head), 4, (0, 0, 255), -1)
        cv2.line(output_img_zoomed, (box_center_x, box_center_y), (cX_head, cY_head), (0, 255, 255), 2)

        direction = "right arrow" if cX_head > box_center_x else "left arrow"
            
        cv2.putText(output_img_zoomed, direction, (box_center_x + 15, box_center_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(f"Direct Method Result: {direction}")
    else:
        print("No arrows found in the wide shot.")

    # --- FINAL DISPLAY ---
    cv2.imshow("Pipeline 1: Final Output", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()