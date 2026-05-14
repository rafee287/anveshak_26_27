import cv2
import numpy as np

# gray - 50 -- r2,l3-1,l1-1
# gray - 100 -- r3-1, r1

# Loading the image
img_orig = cv2.imread("arrow_right.jpg")

if img_orig is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Original Image
    # cv2.imshow("Debug 1: Original Image", img_orig)
    # cv2.waitKey(0)

    # zoom for 1.5x just for slight improvement
    height, width = img_orig.shape[:2]
    new_height = int(1.5 * height)
    new_width = int(1.5 * width)
    
    img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Zoomed image
    # cv2.imshow("Debug 2: Zoomed 1.5x", img_zoomed)
    # cv2.waitKey(0)

    # Converting to grayscale
    gray_zoomed = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

    # Grayscale
    # cv2.imshow("Debug 3: Grayscale Zoomed", gray_zoomed)
    # cv2.waitKey(0)

    # BINARY THRESHOLDING & CLOSING ---
    _, binary_zoomed = cv2.threshold(gray_zoomed, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Checking the Raw Threshold Mask (Before Closing)
    # cv2.imshow("Debug 4: Raw Threshold Mask", binary_zoomed)
    # cv2.waitKey(0)

    # closing any almost closed shapes by morphing pixels
    kernel = np.ones((1, 1), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_zoomed, cv2.MORPH_CLOSE, kernel)

    # Debug Step 4.5: Check the Cleaned Mask (After Closing)
    # cv2.imshow("Debug 4.5: Cleaned Mask (After Closing)", cleaned_mask)
    # cv2.waitKey(0)

    # Finding all contours
    contours_zoomed, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering for Edges, Area, Length, and Straight Lines
    valid_contours = []
    for cnt in contours_zoomed:
        
        # Get the bounding box of the contour (x, y is top-left corner; w, h is width and height)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Checking if the box touches the extreme edges of the zoomed image 
        # using a 1-pixel safety margin just in case
        touches_edge = (x <= 1 or y <= 1 or (x + w) >= (new_width - 1) or (y + h) >= (new_height - 1))
        
        # only proceed if the contour is safely completely inside the frame
        if not touches_edge:
            
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            
            # Keep your exact area and length values
            if area > 50 and length < 800: #and area < 2000:
                
                # --- The Geometry Filter ---
                blank_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
                cv2.drawContours(blank_mask, [cnt], -1, 255, 1)
                
                lines = cv2.HoughLinesP(blank_mask, 1, np.pi/180, threshold=15, minLineLength=15, maxLineGap=5)
                
                if lines is not None:
                    valid_contours.append(cnt)

    output_img_zoomed = img_zoomed.copy()
    cv2.drawContours(output_img_zoomed, valid_contours, -1, (0, 255, 0), 2)

    # verifyig the filtered contours
    # cv2.imshow("Debug 6: Filtered Contours Drawn", output_img_zoomed)
    # cv2.waitKey(0)

    # Draw dot and print area of the largest valid contour 
    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # --- THE DISTANCE TRANSFORM & HYBRID METHOD ---
        
        # 1. Get the TRUE Geometric Center using the rotated rectangle
        rect = cv2.minAreaRect(largest_contour)
        box_center_x = int(rect[0][0])
        box_center_y = int(rect[0][1])
        
        # Draw the cyan bounding box so we can visually verify the geometric center
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(output_img_zoomed, [box], 0, (255, 255, 0), 2)

        # 2. Get the Arrowhead Center using the Distance Transform
        isolated_arrow_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
        cv2.drawContours(isolated_arrow_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # DIST_L2 is standard Euclidean distance (straight line)
        dist_transform = cv2.distanceTransform(isolated_arrow_mask, cv2.DIST_L2, 5)

        # Debug Step 7a: Look at the "Heatmap"! 
        # (Uncomment the two lines below to see how the code finds the thickest part)
        # cv2.imshow("Debug 7a: Distance Transform Heatmap", cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX))
        # cv2.waitKey(0)

        # 3. Find the exact peak of the mountain!
        # minMaxLoc searches the heatmap and gives us the exact (X, Y) coordinates of the highest number
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

        # That maximum location IS the exact center of your arrowhead!
        cX_head = max_loc[0]
        cY_head = max_loc[1]

        # Draw a Green dot for the Geometric Center (Bounding Box)
        cv2.circle(output_img_zoomed, (box_center_x, box_center_y), 5, (0, 255, 0), -1)
        
        # Draw a Red dot for the Arrowhead Center
        cv2.circle(output_img_zoomed, (cX_head, cY_head), 5, (0, 0, 255), -1)
        
        # Draw a line connecting them
        cv2.line(output_img_zoomed, (box_center_x, box_center_y), (cX_head, cY_head), (0, 255, 255), 2)

        # The Logic: Compare Arrowhead X to Geometric Center X
        if cX_head > box_center_x:
            text_to_display = "right arrow"
        else:
            text_to_display = "left arrow"
            
        text_position = (box_center_x + 15, box_center_y - 15)
        cv2.putText(output_img_zoomed, text_to_display, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        print(f"Direction evaluated. Result: {text_to_display}")
    else:
        print("No shapes matching all criteria (Edges, Area, Length, Geometry) were found.")

    # Final Display
    cv2.imshow("Final Zoomed Output", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()