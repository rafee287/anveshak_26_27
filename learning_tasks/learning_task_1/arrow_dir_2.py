import cv2
import numpy as np

# gray - 100 -- r3-1, r1-1 
# gray - 50 -- r2-1

# Loading the image
img_orig = cv2.imread("arrow_right_3.jpg")

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
    _, binary_zoomed = cv2.threshold(gray_zoomed, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Checking the Raw Threshold Mask (Before Closing)
    cv2.imshow("Debug 4: Raw Threshold Mask", binary_zoomed)
    cv2.waitKey(0)

    # closing any almost closed shapes by morphing pixels
    # kernel = np.ones((2,2), np.uint8)
    # cleaned_mask = cv2.morphologyEx(binary_zoomed, cv2.MORPH_CLOSE, kernel)

    # Debug Step 4.5: Check the Cleaned Mask (After Closing)
    # cv2.imshow("Debug 4.5: Cleaned Mask (After Closing)", cleaned_mask)
    # cv2.waitKey(0)

    # Finding all contours
    # contours_zoomed, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_zoomed, _ = cv2.findContours(binary_zoomed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            if area > 50 and length < 350: #and area < 2000:
                
                # --- The Geometry Filter ---
                blank_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
                cv2.drawContours(blank_mask, [cnt], -1, 255, 1)
                
                lines = cv2.HoughLinesP(blank_mask, 1, np.pi/180, threshold=15, minLineLength=15, maxLineGap=5)
                
                if lines is not None:
                    valid_contours.append(cnt)

    output_img_zoomed = img_zoomed.copy()
    cv2.drawContours(output_img_zoomed, valid_contours, -1, (0, 255, 0), 2)

    # verifyig the filtered contours
    cv2.imshow("Debug 6: Filtered Contours Drawn", output_img_zoomed)
    cv2.waitKey(0)

    # Draw dot and print area of the largest valid contour 
    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)

        M_zoomed = cv2.moments(largest_contour)

        if M_zoomed["m00"] != 0:
            cX_zoomed = int(M_zoomed["m10"] / M_zoomed["m00"])
            cY_zoomed = int(M_zoomed["m01"] / M_zoomed["m00"])

            # --- VECTOR TRACKING (EROSION) METHOD FOR DIRECTION ---
            
            # 1. Create a blank canvas and draw ONLY our final validated arrow (filled in white)
            isolated_arrow_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
            cv2.drawContours(isolated_arrow_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # 2. Aggressively Erode this mask to destroy the thin shaft
            erode_kernel = np.ones((1, 1), np.uint8)
            eroded_mask = cv2.erode(isolated_arrow_mask, erode_kernel, iterations=0)
            
            # Debug Step 7a: View the eroded mask to ensure the shaft is gone!
            cv2.imshow("Debug 7a: Eroded Arrowhead", eroded_mask)
            cv2.waitKey(0)

            # 3. Find the contour of the surviving Arrowhead
            head_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(head_contours) > 0:
                head_contour = max(head_contours, key=cv2.contourArea)
                M_head = cv2.moments(head_contour)

                if M_head["m00"] != 0:
                    cX_head = int(M_head["m10"] / M_head["m00"])
                    cY_head = int(M_head["m01"] / M_head["m00"])

                    # Draw the red dot for the WHOLE arrow
                    cv2.circle(output_img_zoomed, (cX_zoomed, cY_zoomed), 5, (0, 0, 255), -1)
                    # Draw a green dot for the ARROWHEAD
                    cv2.circle(output_img_zoomed, (cX_head, cY_head), 5, (0, 255, 0), -1)
                    
                    # Draw a Cyan line connecting them to visualize your vector!
                    cv2.line(output_img_zoomed, (cX_zoomed, cY_zoomed), (cX_head, cY_head), (255, 255, 0), 2)

                    # The Bulletproof Logic: The Head is ALWAYS in the direction the arrow points!
                    if cX_head > cX_zoomed:
                        text_to_display = "right arrow"
                    else:
                        text_to_display = "left arrow"
            else:
                text_to_display = "Error: Erosion destroyed the shape"
            # -------------------------------------------------------------
            
            text_position = (cX_zoomed + 15, cY_zoomed - 15)
            cv2.putText(output_img_zoomed, text_to_display, text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            print(f"Largest shape center found at: X={cX_zoomed}, Y={cY_zoomed}. Direction printed on image.")
    else:
        print("No shapes matching all criteria (Edges, Area, Length, Geometry) were found.")

    # Final Display
    cv2.imshow("Final Zoomed Output", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()