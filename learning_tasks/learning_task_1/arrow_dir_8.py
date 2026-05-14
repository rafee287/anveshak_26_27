import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_arrow(img_path):
    # Loading the image
    img_orig = cv2.imread(img_path)

    if img_orig is None:
        # If the image isn't found, return a blank black image with an error text
        blank_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank_img, "Image Not Found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return blank_img, "File Missing"

    # --- DEBUG 1: Original Image ---
    # cv2.imshow("Debug 1: Original Image", img_orig)
    # cv2.waitKey(0)

    # --- CROPPING LOGIC ---
    # Top 30%, Left 5%, Right 5% of the ORIGINAL image (No Zoom, No Bottom Crop)
    height, width = img_orig.shape[:2]
    y_start = int(height * 0.30)
    y_end = height  
    x_start = int(width * 0.05)
    x_end = int(width * 0.95)   

    img_cropped = img_orig[y_start:y_end, x_start:x_end]

    # --- DEBUG 2: Cropped Region of Interest ---
    # cv2.imshow("Debug 2: Cropped Image", img_cropped)
    # cv2.waitKey(0)

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
    # cv2.imshow("Debug 3: Adaptive Threshold Mask", binary_cropped)
    # cv2.waitKey(0)
    
    # Morphological closing removed.

    # --- FIND CONTOURS (Using RETR_TREE) ---
    contours_cropped, _ = cv2.findContours(binary_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    
    for cnt in contours_cropped:
        x, y, w, h = cv2.boundingRect(cnt)
        touches_edge = (x <= 10 or y <= 10 or (x + w) >= (crop_width - 10) or (y + h) >= (crop_height - 10))
        
        if not touches_edge and h > 0:
            area = cv2.contourArea(cnt)
            aspect_ratio = float(w) / h
            
            # Area > 50, Aspect Ratio between 1.0 and 4.0 (Length limit removed)
            if area > 50 and (1.0 < aspect_ratio < 4.0): 
                
                # --- The Geometry Filter ---
                blank_mask = np.zeros(gray_cropped.shape, dtype=np.uint8)
                cv2.drawContours(blank_mask, [cnt], -1, 255, 1)
                lines = cv2.HoughLinesP(blank_mask, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=5)
                
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
                    
                    # --- DEBUG 4: Halo Testing Area ---
                    # cv2.imshow("Debug 4: Halo Ring Testing Area", halo_mask)
                    # cv2.waitKey(0)

                    if mean_halo_color > 200:
                        valid_contours.append(cnt)

    output_img_cropped = img_cropped.copy()
    cv2.drawContours(output_img_cropped, valid_contours, -1, (0, 255, 0), 2)
    
    detected_direction = "Not Found"

    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)

        M_cropped = cv2.moments(largest_contour)

        if M_cropped["m00"] != 0:
            cX_cropped = int(M_cropped["m10"] / M_cropped["m00"])
            cY_cropped = int(M_cropped["m01"] / M_cropped["m00"])

            leftmost = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
            
            dist_to_left = np.sqrt((cX_cropped - leftmost[0])**2 + (cY_cropped - leftmost[1])**2)
            dist_to_right = np.sqrt((cX_cropped - rightmost[0])**2 + (cY_cropped - rightmost[1])**2)

            isolated_arrow_mask = np.zeros(gray_cropped.shape, dtype=np.uint8)
            cv2.drawContours(isolated_arrow_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            dist_transform = cv2.distanceTransform(isolated_arrow_mask, cv2.DIST_L2, 5)
            
            # --- DEBUG 5: Distance Transform ---
            # cv2.imshow("Debug 5: Distance Transform Heatmap", cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX))
            # cv2.waitKey(0)

            cv2.circle(output_img_cropped, (cX_cropped, cY_cropped), 5, (0, 0, 255), -1)
            cv2.circle(output_img_cropped, leftmost, 4, (255, 0, 0), -1)
            cv2.circle(output_img_cropped, rightmost, 4, (255, 0, 0), -1)
            
            if dist_to_right < dist_to_left:
                detected_direction = "right arrow"
            else:
                detected_direction = "left arrow"
            
            text_position = (cX_cropped + 15, cY_cropped - 15)
            cv2.putText(output_img_cropped, detected_direction, text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
    # Return the processed image and the final text logic for the Matplotlib title
    return output_img_cropped, detected_direction


# ==========================================
# MAIN BATCH PROCESSING & MATPLOTLIB PLOTTING
# ==========================================

print("Starting Batch Processing...")

# --- UPDATED: The 6 files you want to process ---
image_files = [
    "arrow_left.jpg", "arrow_right.jpg", 
    "arrow_left_2.jpg", "arrow_right_2.jpg", 
    "arrow_left_3.jpg", "arrow_right_3.jpg"
]

# Set up a 3x2 grid in Matplotlib
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.canvas.manager.set_window_title('Arrow Detection Pipeline Results')

# Flatten the axes array to easily iterate over it
axes = axes.flatten()

for i, filename in enumerate(image_files):
    print(f"Processing: {filename}...")
    
    # Pass the image through our pipeline
    processed_img, direction = process_arrow(filename)
    
    # OpenCV uses BGR, but Matplotlib requires RGB format to display colors correctly
    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # Plot the image on the corresponding grid square
    axes[i].imshow(img_rgb)
    axes[i].set_title(f"File: {filename}\nResult: {direction}", fontweight="bold")
    axes[i].axis('off')  # Hide the X and Y axis tick marks

# Adjust layout so titles don't overlap and show the final plot
plt.tight_layout()
print("Rendering Grid...")
plt.show()