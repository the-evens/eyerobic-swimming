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
    prev_avg_line_endpoints = None # For temporal smoothing
    SMOOTHING_FACTOR = 0.1         # Weight for the current frame's line (decreased from 0.2 for EXTRA smooth)
    MAX_ALLOWED_JUMP_RATIO = 0.25  # Max jump distance as a ratio of image width
    
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
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=100, maxLineGap=30)

            if lines is not None:
                line_candidates = []
                for line_segment in lines:
                    x1, y1, x2, y2 = line_segment[0]
                    avg_x_segment = (x1 + x2) / 2.0
                    distance_to_center = abs(avg_x_segment - image_center_x)
                    angle_rad = np.arctan2(y2 - y1, x2 - x1)
                    line_candidates.append({'segment': (x1, y1, x2, y2), 'distance': distance_to_center, 'avg_x': avg_x_segment, 'angle': angle_rad})

                if line_candidates:
                    line_candidates.sort(key=lambda item: item['distance'])
                    
                    selected_lines_data = []
                    if line_candidates: # Ensure there is at least one candidate
                        selected_lines_data.append(line_candidates[0]) # Add the closest line
                        
                        # Try to find a second, distinct line
                        if len(selected_lines_data) == 1 and len(line_candidates) > 1:
                            line1_data = selected_lines_data[0]
                            
                            angle_similarity_threshold_rad = 0.1  # If acute angle diff < this, angles are "similar"
                            min_avg_x_sep_if_angles_similar = image_width * 0.20
                            min_avg_x_sep_if_angles_different = image_width * 0.05

                            for candidate_data in line_candidates[1:]:
                                # Calculate acute angle difference between the two lines
                                norm_angle1 = np.mod(line1_data['angle'], np.pi)
                                norm_angle_candidate = np.mod(candidate_data['angle'], np.pi)
                                angle_diff_rad = abs(norm_angle1 - norm_angle_candidate)
                                min_acute_angle_diff = min(angle_diff_rad, np.pi - angle_diff_rad)

                                # Determine required x-separation based on angle similarity
                                required_x_separation = min_avg_x_sep_if_angles_similar
                                if min_acute_angle_diff >= angle_similarity_threshold_rad: # Angles are different enough
                                    required_x_separation = min_avg_x_sep_if_angles_different

                                avg_x_dist = abs(candidate_data['avg_x'] - line1_data['avg_x'])

                                if avg_x_dist >= required_x_separation:
                                    selected_lines_data.append(candidate_data)
                                    break # Found our second line

                    extended_lines_endpoints = []
                    for line_data in selected_lines_data: # Iterate over the selected distinct lines (at most 2)
                        x1, y1, x2, y2 = line_data['segment']
                        pt1_ext, pt2_ext = None, None

                        if abs(x1 - x2) < 1e-6: # Vertical segment
                            pt1_ext = (x1, 0)
                            pt2_ext = (x1, image_height - 1)
                        elif abs(y1 - y2) < 1e-6: # Horizontal segment
                            cv2.line(output_lines_on_thresh, (x1, y1), (x2, y2), (0, 255, 255), 8) # Yellow
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
                                cv2.line(output_lines_on_thresh, valid_pt1, valid_pt2, (0, 255, 0), 8) # Green
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

                        current_avg_pt1 = (int(round(avg_pt1_x)), int(round(avg_pt1_y)))
                        current_avg_pt2 = (int(round(avg_pt2_x)), int(round(avg_pt2_y)))
                        
                        display_pt1, display_pt2 = current_avg_pt1, current_avg_pt2

                        if prev_avg_line_endpoints is not None:
                            prev_pt1, prev_pt2 = prev_avg_line_endpoints
                            
                            # Calculate midpoint distance for jerk detection
                            current_mid_x = (current_avg_pt1[0] + current_avg_pt2[0]) / 2.0
                            current_mid_y = (current_avg_pt1[1] + current_avg_pt2[1]) / 2.0
                            prev_mid_x = (prev_pt1[0] + prev_pt2[0]) / 2.0
                            prev_mid_y = (prev_pt1[1] + prev_pt2[1]) / 2.0
                            jump_distance = np.sqrt((current_mid_x - prev_mid_x)**2 + (current_mid_y - prev_mid_y)**2)
                            max_jump = image_width * MAX_ALLOWED_JUMP_RATIO

                            if jump_distance < max_jump:
                                # Apply smoothing if jump is within acceptable limits
                                smooth_pt1_x = SMOOTHING_FACTOR * current_avg_pt1[0] + (1 - SMOOTHING_FACTOR) * prev_pt1[0]
                                smooth_pt1_y = SMOOTHING_FACTOR * current_avg_pt1[1] + (1 - SMOOTHING_FACTOR) * prev_pt1[1]
                                smooth_pt2_x = SMOOTHING_FACTOR * current_avg_pt2[0] + (1 - SMOOTHING_FACTOR) * prev_pt2[0]
                                smooth_pt2_y = SMOOTHING_FACTOR * current_avg_pt2[1] + (1 - SMOOTHING_FACTOR) * prev_pt2[1]
                                display_pt1 = (int(round(smooth_pt1_x)), int(round(smooth_pt1_y)))
                                display_pt2 = (int(round(smooth_pt2_x)), int(round(smooth_pt2_y)))
                            # Else (jump_distance >= max_jump): use current_avg_pt1, current_avg_pt2 directly (already set)
                            # This allows the line to adapt quickly to large, potentially correct shifts.
                        
                        cv2.line(output_lines_on_thresh, display_pt1, display_pt2, (255, 0, 0), 8) # Blue
                        prev_avg_line_endpoints = (current_avg_pt1, current_avg_pt2) # Store unsmoothed for next iter
                    else:
                        # If no two valid lines found, reset smoothing history
                        prev_avg_line_endpoints = None
            
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