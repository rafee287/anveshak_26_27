import cv2
import numpy as np

img_orig = cv2.imread("arrow_left_3.jpg")

if img_orig is None:
    print("Error: Could not load image.")
else:
    height, width = img_orig.shape[:2]
    new_height = int(1.5 * height)
    new_width = int(1.5 * width)
    img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    gray_zoomed = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)
    output_img_zoomed = img_zoomed.copy()

    # ==========================================
    # STAGE 1: FIND THE WHITE POST
    # ==========================================
    
    # Standard thresholding for bright white objects (ignoring dirt/trees)
    # The sky will be picked up, but we filter it out using Area and Aspect Ratio
    _, white_mask = cv2.threshold(gray_zoomed, 200, 255, cv2.THRESH_BINARY)
    
    post_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    post_roi = None
    post_coords = None

    for cnt in post_contours:
        area = cv2.contourArea(cnt)
        
        # Filter: It must be decently sized, but not the whole sky
        if 500 < area < 50000: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter: A post is a vertical rectangle (Height is greater than Width)
            aspect_ratio = float(h) / w
            if aspect_ratio > 1.5: 
                
                # WE FOUND THE POST! 
                # Let's crop the original zoomed image to exactly these coordinates
                post_roi = img_zoomed[y:y+h, x:x+w]
                post_coords = (x, y, w, h)
                
                # Draw a yellow box around the post on the main image
                cv2.rectangle(output_img_zoomed, (x, y), (x+w, y+h), (0, 255, 255), 2)
                break # Stop searching once we find the best post

    # ==========================================
    # STAGE 2: FIND ARROW INSIDE THE POST (ROI)
    # ==========================================
    
    if post_roi is not None:
        print("Signpost found! Scanning Region of Interest...")
        
        # Convert just the tiny cropped post to grayscale
        roi_gray = cv2.cvtColor(post_roi, cv2.COLOR_BGR2GRAY)
        
        # Because it's cropped, the background is perfectly white!
        # Otsu will work flawlessly here.
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        roi_contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(roi_contours) > 0:
            # We can just assume the biggest black shape on the white post is the arrow
            arrow_cnt = max(roi_contours, key=cv2.contourArea)
            
            # Ensure it's not just a speck of dirt
            if cv2.contourArea(arrow_cnt) > 20: 
                
                # Distance Transform & Direction Logic (Running locally inside the ROI)
                rect = cv2.minAreaRect(arrow_cnt)
                box_center_x = int(rect[0][0])
                
                isolated_arrow_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
                cv2.drawContours(isolated_arrow_mask, [arrow_cnt], -1, 255, thickness=cv2.FILLED)
                dist_transform = cv2.distanceTransform(isolated_arrow_mask, cv2.DIST_L2, 5)
                _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
                cX_head = max_loc[0]
                
                direction = "right arrow" if cX_head > box_center_x else "left arrow"
                
                # Print the result above the yellow box on the MAIN image
                post_x, post_y, _, _ = post_coords
                cv2.putText(output_img_zoomed, direction, (post_x, post_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                print(f"ROI Method Result: {direction}")
                
                # Show the cropped ROI as a separate window so you can see what the computer sees!
                cv2.imshow("Cropped ROI View", post_roi)
                cv2.imshow("ROI Binary Mask", roi_binary)

    else:
        print("Could not find a white vertical post in the image.")

    cv2.imshow("Pipeline 2: Main Camera View", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()