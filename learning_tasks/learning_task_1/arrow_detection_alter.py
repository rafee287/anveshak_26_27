import cv2
import numpy as np


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
    cv2.imshow("Debug 4.5: Cleaned Mask (After Closing)", cleaned_mask)
    cv2.waitKey(0)

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
    # cv2.imshow("Debug 6: Filtered Contours Drawn", output_img_zoomed)
    # cv2.waitKey(0)

    # Draw dot and print area of the largest valid contour 
    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        x_0, y_0 , w, h = cv2.boundingRect(largest_contour)
        # Grab the exact area value to print
        max_area = cv2.contourArea(largest_contour)

        M_zoomed = cv2.moments(largest_contour)

        if M_zoomed["m00"] != 0:
            cX_zoomed = int(M_zoomed["m10"] / M_zoomed["m00"])
            cY_zoomed = int(M_zoomed["m01"] / M_zoomed["m00"])

            # Draw the red dot at the calculated center of mass (of sorts) and also at the actual centre of 
            cv2.circle(output_img_zoomed, (cX_zoomed, cY_zoomed), 5, (0, 0, 255), -1)
            cv2.circle(output_img_zoomed, (int(x_0+w/2), int(y_0+h/2)), 5, (0, 255, 0), -1)
            cv2.rectangle(output_img_zoomed,(x_0,y_0),(x_0+w,y_0+h),(255,0,0),2)
            
            if cX_zoomed > x_0+w/2:
                text_to_display = f"right arrow"
            else:
                text_to_display = f"left arrow"
            
            text_position = (cX_zoomed + 15, cY_zoomed - 15)
            cv2.putText(output_img_zoomed, text_to_display, text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            print(f"Largest shape center found at: X={cX_zoomed}, Y={cY_zoomed}. Area printed on image.")
    else:
        print("No shapes matching all criteria (Edges, Area, Length, Geometry) were found.")

    # Final Display
    cv2.imshow("Final Zoomed Output", output_img_zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# import cv2
# import numpy as np


# # Loading the image
# img_orig = cv2.imread("arrow_right_3.jpg")

# if img_orig is None:
#     print("Error: Could not load image. Check the file path.")
# else:
#     # Original Image
#     # cv2.imshow("Debug 1: Original Image", img_orig)
#     # cv2.waitKey(0)

#     # zoom for 1.5x just for slight improvement
#     height, width = img_orig.shape[:2]
#     new_height = int(1.5 * height)
#     new_width = int(1.5 * width)
    
#     img_zoomed = cv2.resize(img_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

#     # Zoomed image
#     # cv2.imshow("Debug 2: Zoomed 1.5x", img_zoomed)
#     # cv2.waitKey(0)

#     # Converting to grayscale
#     gray_zoomed = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

#     # Grayscale
#     # cv2.imshow("Debug 3: Grayscale Zoomed", gray_zoomed)
#     # cv2.waitKey(0)

#     # BINARY THRESHOLDING & CLOSING ---
#     _, binary_zoomed = cv2.threshold(gray_zoomed, 100, 255, cv2.THRESH_BINARY_INV)
    
#     # Checking the Raw Threshold Mask (Before Closing)
#     # cv2.imshow("Debug 4: Raw Threshold Mask", binary_zoomed)
#     # cv2.waitKey(0)

#     # closing any almost closed shapes by morphing pixels
#     kernel = np.ones((1, 1), np.uint8)
#     cleaned_mask = cv2.morphologyEx(binary_zoomed, cv2.MORPH_CLOSE, kernel)

#     # Debug Step 4.5: Check the Cleaned Mask (After Closing)
#     # cv2.imshow("Debug 4.5: Cleaned Mask (After Closing)", cleaned_mask)
#     # cv2.waitKey(0)

#     # Finding all contours
#     contours_zoomed, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filtering for Edges, Area, Length, and Straight Lines
#     valid_contours = []
#     for cnt in contours_zoomed:
        
#         # Get the bounding box of the contour (x, y is top-left corner; w, h is width and height)
#         x, y, w, h = cv2.boundingRect(cnt)
        
#         # Checking if the box touches the extreme edges of the zoomed image 
#         # using a 1-pixel safety margin just in case
#         touches_edge = (x <= 1 or y <= 1 or (x + w) >= (new_width - 1) or (y + h) >= (new_height - 1))
        
#         # only proceed if the contour is safely completely inside the frame
#         if not touches_edge:
            
#             area = cv2.contourArea(cnt)
#             length = cv2.arcLength(cnt, True)
            
#             # Keep your exact area and length values
#             if area > 50 and length < 350: #and area < 2000:
                
#                 # --- The Geometry Filter ---
#                 blank_mask = np.zeros(gray_zoomed.shape, dtype=np.uint8)
#                 cv2.drawContours(blank_mask, [cnt], -1, 255, 1)
                
#                 lines = cv2.HoughLinesP(blank_mask, 1, np.pi/180, threshold=15, minLineLength=15, maxLineGap=5)
                
#                 if lines is not None:
#                     valid_contours.append(cnt)

#     output_img_zoomed = img_zoomed.copy()
#     cv2.drawContours(output_img_zoomed, valid_contours, -1, (0, 255, 0), 2)

#     # verifyig the filtered contours
#     # cv2.imshow("Debug 6: Filtered Contours Drawn", output_img_zoomed)
#     # cv2.waitKey(0)

#     # Draw dot and print area of the largest valid contour 
#     if len(valid_contours) > 0:
#         largest_contour = max(valid_contours, key=cv2.contourArea)
        
#         rect = cv2.minAreaRect(largest_contour)
#         box_center_x = rect[0][0] 
#         box_center_y = rect[0][1]

#         # --- CHANGES ADDED HERE: Calculate and draw the rotated bounding box ---
#         # 1. Get the 4 floating point corners of the rectangle
#         box = cv2.boxPoints(rect)
#         # 2. Convert the coordinates to integers so OpenCV can draw them
#         box = np.int32(box)
#         # 3. Draw the box outline in Cyan. (We put 'box' in brackets because drawContours expects a list of shapes)
#         cv2.drawContours(output_img_zoomed, [box], 0, (255, 255, 0), 2)


#         M_zoomed = cv2.moments(largest_contour)

#         if M_zoomed["m00"] != 0:
#             cX_zoomed = int(M_zoomed["m10"] / M_zoomed["m00"])
#             cY_zoomed = int(M_zoomed["m01"] / M_zoomed["m00"])

#             # Draw the red dot at the calculated center of mass (of sorts) and also at the actual centre of 
#             cv2.circle(output_img_zoomed, (cX_zoomed, cY_zoomed), 5, (0, 0, 255), -1)
#             cv2.circle(output_img_zoomed, (int(box_center_x), int(box_center_y)), 5, (0, 255, 0), -1)
            
#             if cX_zoomed > box_center_x:
#                 text_to_display = f"right arrow"
#             else:
#                 text_to_display = f"left arrow"
            
#             text_position = (cX_zoomed + 15, cY_zoomed - 15)
#             cv2.putText(output_img_zoomed, text_to_display, text_position, 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
#             print(f"Largest shape center found at: X={cX_zoomed}, Y={cY_zoomed}. Area printed on image.")
#     else:
#         print("No shapes matching all criteria (Edges, Area, Length, Geometry) were found.")

#     # Final Display
#     cv2.imshow("Final Zoomed Output", output_img_zoomed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()