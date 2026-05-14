import cv2
import numpy as np

# --- Step 1: Load the image ---
img_orig = cv2.imread("arrow_right.jpg")

if img_orig is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Debug Step 1: Original Image
    # cv2.imshow("Debug 1: Original Image", img_orig)
    # cv2.waitKey(0)

    # --- Step 2: THE ZOOM (1.5x) ---
    height, width = img_orig.shape[:2]
    new_height = int(1.5 * height)
    new_width = int(1.5 * width)
    
    # Resize the image. All math from here on uses this zoomed version.
    img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Debug Step 2: Check the Zoom
    # cv2.imshow("Debug 2: Zoomed 1.5x", img_zoomed)
    # cv2.waitKey(0)

    # --- Step 3: Convert to grayscale ---
    gray_zoomed = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

    # Debug Step 3: Check Grayscale
    # cv2.imshow("Debug 3: Grayscale Zoomed", gray_zoomed)
    # cv2.waitKey(0)

    # --- Step 4: Perform Canny edge detection ---
    # UPDATED: High threshold to grab only the sharpest, highest-contrast edges
    canny_zoomed = cv2.Canny(gray_zoomed, 250, 300)

    # Debug Step 4a: Check the Edge Map
    # cv2.imshow("Debug 4: Canny Edges Zoomed", canny_zoomed)
    # cv2.waitKey(0)

    # --- NEW STEP: Morphological Closing to seal broken edges ---
    # 1. Create a "Kernel" (A 5x5 block of pixels)
    # The size of this block determines how wide of a gap it can jump. 
    # A 5x5 kernel will easily seal a 1 to 4 pixel break.
    kernel = np.ones((1, 1), np.uint8)
    
    # 2. Apply the Closing operation
    # This dilates (expands) the white edges to touch each other, 
    # then erodes (shrinks) them back down so the shape doesn't get distorted.
    closed_edges = cv2.morphologyEx(canny_zoomed, cv2.MORPH_CLOSE, kernel)

    # Debug Step 4a: Check the Edge Map
    cv2.imshow("Debug 4: Canny Edges Zoomed", closed_edges)
    cv2.waitKey(0)

    # --- Step 5: Find ALL contours ---
    contours_zoomed, _ = cv2.findContours(canny_zoomed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # --- NEW LOGIC: Filter for Area AND Outline Length ---
    valid_contours = []
    for cnt in contours_zoomed:
        # Measure the area trapped inside
        area = cv2.contourArea(cnt)
        
        # Measure the length of the outline (Perimeter)
        # The 'True' flag tells OpenCV to assume the shape is closed when measuring
        length = cv2.arcLength(cnt, True)
        
        # The Filter: Must have area > 50 AND length < 500
        if area > 50 and length < 1000:
            valid_contours.append(cnt)

    # Create a copy of the zoomed image to draw the output on
    output_img_zoomed = img_zoomed.copy()
    
    # Draw ONLY the valid, filtered contours on the zoomed image in Green
    cv2.drawContours(output_img_zoomed, valid_contours, -1, (0, 255, 0), 2)

    # Debug Step 5: Verify that only shapes passing both filters got outlined
    cv2.imshow("Debug 5: Filtered Contours Drawn", output_img_zoomed)
    cv2.waitKey(0)

    # --- Step 6: Draw a red dot in the center of the largest valid contour ---
    if len(valid_contours) > 0:
        # Find the single largest contour from our newly filtered list
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Calculate moments to find the center
        M_zoomed = cv2.moments(largest_contour)

        # Prevent division by zero
        if M_zoomed["m00"] != 0:
            cX_zoomed = int(M_zoomed["m10"] / M_zoomed["m00"])
            cY_zoomed = int(M_zoomed["m01"] / M_zoomed["m00"])

            # Draw the red dot at the calculated center
            cv2.circle(output_img_zoomed, (cX_zoomed, cY_zoomed), 5, (0, 0, 255), -1)
            
            print(f"Largest shape center (zoomed) found at: X={cX_zoomed}, Y={cY_zoomed}")
    else:
        print("No shapes matching the area and length criteria were found.")

    # Final Display (Always runs)
    cv2.imshow("Final Zoomed Output", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()