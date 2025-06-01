import cv2
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    cv2.namedWindow('Original')
    cv2.namedWindow('BW Post Threshold')
    cv2.namedWindow('Lines on Thresh')
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display original frame
            cv2.imshow('Original', frame)
            
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur (kernel size 5x5, sigma=1)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)
            # Apply binary thresholding (threshold at 70)
            _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
            cv2.imshow('BW Post Threshold', thresh)

            # --- Start Line Detection Integration ---
            output_lines_on_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            image_height, image_width = thresh.shape[:2]
            image_center_x = image_width / 2.0

            edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

            if lines is not None:
                line_candidates = []
                for line_segment in lines:
                    x1, y1, x2, y2 = line_segment[0]
                    avg_x_segment = (x1 + x2) / 2.0
                    distance_to_center = abs(avg_x_segment - image_center_x)
                    line_candidates.append({'segment': (x1, y1, x2, y2), 'distance': distance_to_center})

                if line_candidates:
                    line_candidates.sort(key=lambda item: item['distance'])
                    
                    extended_lines_endpoints = []
                    num_to_process = min(2, len(line_candidates))

                    for i in range(num_to_process):
                        line_data = line_candidates[i]
                        x1, y1, x2, y2 = line_data['segment']
                        pt1_ext, pt2_ext = None, None

                        if abs(x1 - x2) < 1e-6: # Vertical segment
                            pt1_ext = (x1, 0)
                            pt2_ext = (x1, image_height - 1)
                        elif abs(y1 - y2) < 1e-6: # Horizontal segment
                            cv2.line(output_lines_on_thresh, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                            extended_lines_endpoints.append(None)
                            continue
                        else: # Slanted segment
                            slope = (y2 - y1) / (x2 - x1 + 1e-6) # Add epsilon to avoid division by zero
                            intercept_b = y1 - slope * x1
                            
                            x_at_top = (0 - intercept_b) / (slope + 1e-6) # Add epsilon for slope near zero
                            x_at_bottom = (image_height - 1 - intercept_b) / (slope + 1e-6)
                            
                            pt1_ext = (int(round(x_at_top)), 0)
                            pt2_ext = (int(round(x_at_bottom)), image_height - 1)
                        
                        if pt1_ext and pt2_ext:
                            # Clip points to be within image boundaries
                            valid_pt1_x = max(0, min(pt1_ext[0], image_width - 1))
                            valid_pt1_y = max(0, min(pt1_ext[1], image_height - 1))
                            valid_pt2_x = max(0, min(pt2_ext[0], image_width - 1))
                            valid_pt2_y = max(0, min(pt2_ext[1], image_height - 1))
                            valid_pt1 = (valid_pt1_x, valid_pt1_y)
                            valid_pt2 = (valid_pt2_x, valid_pt2_y)

                            # Draw if the line has some length after extension and clipping
                            if np.sqrt((valid_pt2[0]-valid_pt1[0])**2 + (valid_pt2[1]-valid_pt1[1])**2) > 10:
                                cv2.line(output_lines_on_thresh, valid_pt1, valid_pt2, (0, 255, 0), 2) # Green
                                extended_lines_endpoints.append((valid_pt1, valid_pt2))
                            else:
                                extended_lines_endpoints.append(None)
                        else:
                            extended_lines_endpoints.append(None)

                    valid_extended_lines = [line for line in extended_lines_endpoints if line is not None]
                    if len(valid_extended_lines) == 2:
                        (l1_pt1, l1_pt2) = valid_extended_lines[0]
                        (l2_pt1, l2_pt2) = valid_extended_lines[1]
                        
                        avg_pt1_x = (l1_pt1[0] + l2_pt1[0]) / 2.0
                        avg_pt1_y = (l1_pt1[1] + l2_pt1[1]) / 2.0 # Should be 0 for lines extended to top
                        
                        avg_pt2_x = (l1_pt2[0] + l2_pt2[0]) / 2.0
                        avg_pt2_y = (l1_pt2[1] + l2_pt2[1]) / 2.0 # Should be image_height-1 for lines extended to bottom

                        avg_pt1 = (int(round(avg_pt1_x)), int(round(avg_pt1_y)))
                        avg_pt2 = (int(round(avg_pt2_x)), int(round(avg_pt2_y)))
                        
                        cv2.line(output_lines_on_thresh, avg_pt1, avg_pt2, (255, 0, 0), 6) # Blue
            
            cv2.imshow('Lines on Thresh', output_lines_on_thresh)
            # --- End Line Detection Integration ---

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 