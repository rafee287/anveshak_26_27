# final solution

import cv2
import numpy as np

print("Starting Arrow Detection Pipeline...")
print("NOTE: Press any key on your keyboard to advance through the Debug windows.\n")

# Loading the image
img_orig = cv2.imread("arrow_left.jpg")

if img_orig is None:
    print("Error: Could not load image. Check the file path.")
else:
    # --- DEBUG 1: Original Image ---
    cv2.imshow("Debug 1: Original Image", img_orig)
    cv2.waitKey(0)

    # --- ZOOM LOGIC ---
    height, width = img_orig.shape[:2]
    new_height = int(1.5 * height)
    new_width = int(1.5 * width)
    
    img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # --- CROPPING LOGIC ---
    # Top 30%, Bottom 20%, Left 5%, Right 5% of the ZOOMED image
    y_start = int(new_height * 0.30)
    y_end = int(new_height * 0.80)  
    x_start = int(new_width * 0.05)
    x_end = int(new_width * 0.95)   

    img_cropped = img_zoomed[y_start:y_end, x_start:x_end]

    # --- DEBUG 2: Cropped Region of Interest ---
    cv2.imshow("Debug 2: Cropped & Zoomed Image (Background Removed)", img_cropped)
    cv2.waitKey(0)

    crop_height, crop_width = img_cropped.shape[:2]
    gray_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    # --- ADAPTIVE BINARY THRESHOLDING ---
    binary_cropped = cv2.adaptiveThreshold(
        gray_cropped, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 
        5
    )
    
    # --- DEBUG 3: Adaptive Threshold ---
    cv2.imshow("Debug 3: Adaptive Threshold Mask (Slicing through shadows)", binary_cropped)
    cv2.waitKey(0)
    
    # closing any almost closed shapes by morphing pixels
    kernel = np.ones((1, 1), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_cropped, cv2.MORPH_CLOSE, kernel)

    # --- DEBUG 4: Cleaned Mask ---
    cv2.imshow("Debug 4: Morphological Closing (Tiny noise holes filled)", cleaned_mask)
    cv2.waitKey(0)

    # Finding all contours
    contours_cropped, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    
    for cnt in contours_cropped:
        x, y, w, h = cv2.boundingRect(cnt)
        touches_edge = (x <= 10 or y <= 10 or (x + w) >= (crop_width - 10) or (y + h) >= (crop_height - 10))
        
        # Added h > 0 to prevent division by zero when calculating aspect ratio
        if not touches_edge and h > 0:
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            
            # Calculate the aspect ratio
            aspect_ratio = float(w) / h
            
            # --- UPDATED FILTER: Area, Length, and Aspect Ratio (3 to 4) ---
            if area > 50 and length < 350 and (1 < aspect_ratio < 4.0): 
                
                # --- The Geometry Filter ---
                blank_mask = np.zeros(gray_cropped.shape, dtype=np.uint8)
                cv2.drawContours(blank_mask, [cnt], -1, 255, 1)
                lines = cv2.HoughLinesP(blank_mask, 1, np.pi/180, threshold=15, minLineLength=15, maxLineGap=5)
                
                if lines is not None:
                    
                    # --- THE "HALO" BACKGROUND CHECK ---
                    single_arrow_mask = np.zeros(gray_cropped.shape, dtype=np.uint8)
                    cv2.drawContours(single_arrow_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    
                    dilate_kernel = np.ones((3, 3), np.uint8)
                    inner_mask = cv2.dilate(single_arrow_mask, dilate_kernel, iterations=1)
                    outer_mask = cv2.dilate(single_arrow_mask, dilate_kernel, iterations=5)
                    
                    halo_mask = cv2.subtract(outer_mask, inner_mask)
                    normal_binary = cv2.bitwise_not(binary_cropped)
                    mean_halo_color = cv2.mean(normal_binary, mask=halo_mask)[0]
                    
                    # --- DEBUG 5: The Halo Mask ---
                    # It will pop up a window for every shape that makes it this far to show you its "ring"
                    cv2.imshow("Debug 5: Halo Ring Testing Area", halo_mask)
                    cv2.waitKey(0)

                    if mean_halo_color > 200:
                        valid_contours.append((cnt, aspect_ratio)) # Storing aspect ratio for the print statement

    output_img_cropped = img_cropped.copy()
    
    # We strip the aspect ratio back out so drawContours just gets the raw contours
    just_contours = [item[0] for item in valid_contours]
    cv2.drawContours(output_img_cropped, just_contours, -1, (0, 255, 0), 2)

    if len(valid_contours) > 0:
        # Find the largest contour out of the valid ones
        largest_valid_item = max(valid_contours, key=lambda item: cv2.contourArea(item[0]))
        largest_contour = largest_valid_item[0]
        final_aspect_ratio = largest_valid_item[1]

        M_cropped = cv2.moments(largest_contour)

        if M_cropped["m00"] != 0:
            cX_cropped = int(M_cropped["m10"] / M_cropped["m00"])
            cY_cropped = int(M_cropped["m01"] / M_cropped["m00"])

            leftmost = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
            
            dist_to_left = np.sqrt((cX_cropped - leftmost[0])**2 + (cY_cropped - leftmost[1])**2)
            dist_to_right = np.sqrt((cX_cropped - rightmost[0])**2 + (cY_cropped - rightmost[1])**2)

            # Distance Transform just for visualization (since you are using extreme points here)
            isolated_arrow_mask = np.zeros(gray_cropped.shape, dtype=np.uint8)
            cv2.drawContours(isolated_arrow_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            dist_transform = cv2.distanceTransform(isolated_arrow_mask, cv2.DIST_L2, 5)
            
            # --- DEBUG 6: Distance Transform ---
            cv2.imshow("Debug 6: Distance Transform Heatmap (Finding the thickest part)", cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX))
            cv2.waitKey(0)

            cv2.circle(output_img_cropped, (cX_cropped, cY_cropped), 5, (0, 0, 255), -1)
            cv2.circle(output_img_cropped, leftmost, 4, (255, 0, 0), -1)
            cv2.circle(output_img_cropped, rightmost, 4, (255, 0, 0), -1)
            
            if dist_to_right < dist_to_left:
                text_to_display = "right arrow"
            else:
                text_to_display = "left arrow"
            
            text_position = (cX_cropped + 15, cY_cropped - 15)
            cv2.putText(output_img_cropped, text_to_display, text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            print(f"\nSUCCESS! Largest shape center found at: X={cX_cropped}, Y={cY_cropped}.")
            print(f"Aspect Ratio: {final_aspect_ratio:.2f}")
            print(f"Direction: {text_to_display}")
    else:
        print("\nFAILURE: No shapes matching all criteria (Edges, Area, Length, Aspect Ratio, Geometry, Background Halo) were found.")

    # --- FINAL DISPLAY ---
    cv2.imshow("Final Cropped Output", output_img_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()