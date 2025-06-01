import cv2
import numpy as np

# Load the image
image = cv2.imread('singular_frame.png')
if image is None:
    print("Error: Could not load image.")
    exit()

output_image = image.copy()
image_height, image_width = image.shape[:2]
image_center_x = image_width / 2

# Image processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=400, maxLineGap=50)

if lines is None:
    print("No line segments found by Hough Transform.")
else:
    line_candidates = []
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        avg_x_segment = (x1 + x2) / 2
        distance_to_center = abs(avg_x_segment - image_center_x)
        line_candidates.append({'segment': (x1, y1, x2, y2), 'distance': distance_to_center})

    if not line_candidates:
        print("No line candidates processed.") # Should not be reached if lines is not None
    else:
        line_candidates.sort(key=lambda item: item['distance'])
        
        extended_lines_endpoints = []
        num_to_process = min(2, len(line_candidates))

        for i in range(num_to_process):
            line_data = line_candidates[i]
            x1, y1, x2, y2 = line_data['segment']

            pt1_ext, pt2_ext = None, None

            if x1 == x2: # Vertical segment
                pt1_ext = (x1, 0)
                pt2_ext = (x1, image_height)
            elif y1 == y2: # Horizontal segment
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 255), 8) # Yellow
                extended_lines_endpoints.append(None) # Cannot be used for averaging as is
                continue
            else: # Slanted segment
                slope = (y2 - y1) / (x2 - x1)
                intercept_b = y1 - slope * x1
                x_at_top = (0 - intercept_b) / slope
                x_at_bottom = (image_height - intercept_b) / slope
                pt1_ext = (int(round(x_at_top)), 0)
                pt2_ext = (int(round(x_at_bottom)), image_height)

            if pt1_ext and pt2_ext:
                cv2.line(output_image, pt1_ext, pt2_ext, (0, 255, 0), 8) # Green
                extended_lines_endpoints.append((pt1_ext, pt2_ext))
            else:
                # This case should ideally not be hit if logic for horizontal/vertical is correct
                extended_lines_endpoints.append(None)

        # Draw the average line if two valid lines were extended
        if len(extended_lines_endpoints) == 2 and all(p is not None for p in extended_lines_endpoints):
            l1_pt1, l1_pt2 = extended_lines_endpoints[0]
            l2_pt1, l2_pt2 = extended_lines_endpoints[1]
            
            avg_pt1_x = (l1_pt1[0] + l2_pt1[0]) / 2
            avg_pt2_x = (l1_pt2[0] + l2_pt2[0]) / 2
            
            avg_pt1 = (int(round(avg_pt1_x)), 0)
            avg_pt2 = (int(round(avg_pt2_x)), image_height)
            cv2.line(output_image, avg_pt1, avg_pt2, (255, 0, 0), 8) # Blue

# Save the output image
cv2.imwrite('lines_detected.png', output_image)
print("Line detection complete. Output saved as lines_detected.png") 