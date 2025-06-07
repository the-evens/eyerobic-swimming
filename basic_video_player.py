import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '1000000'  # Set to a very high value

import cv2
import argparse
import numpy as np
import time  # Add time module for delay

def detect_bubbles(thresh, min_radius=1, max_radius=15, min_circles=3, max_circles=15):
    """Detect bubbles in the thresholded image using Hough Circle Transform"""
    # Apply additional preprocessing to enhance bubble detection
    # Dilate the image to connect nearby white regions
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
    
    # Use Hough Circle Transform with more sensitive parameters
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=3,    # Reduced minimum distance between circles
        param1=15,    # Reduced edge detection sensitivity
        param2=15,    # Reduced accumulator threshold for more sensitive detection
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        # Count how many circles we found
        num_circles = len(circles[0])
        
        # Calculate confidence based on circle properties
        confidence = 0
        for circle in circles[0]:
            x, y, r = circle
            # Higher confidence for circles that are more circular and have good contrast
            if r > 3:  # Reduced minimum radius for higher sensitivity
                confidence += 1
        
        # Require both sufficient number of circles and high confidence
        has_bubbles = num_circles >= min_circles and confidence >= min_circles
        too_many_bubbles = num_circles > max_circles
        
        return has_bubbles, too_many_bubbles, circles[0]
    return False, False, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()
    
    # Set FFmpeg backend parameters
    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size
    
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    # Get the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original video FPS: {original_fps}")
    print("Video is playing at 4x speed")
    print("Controls:")
    print("  ESC: Exit")
    print("  SPACE: Pause/Resume")

    cv2.namedWindow('Original')
    cv2.namedWindow('BW Post Threshold')
    cv2.namedWindow('Lines on Thresh')
    paused = False
    prev_avg_line_endpoints = None # For temporal smoothing
    SMOOTHING_FACTOR = 0.1         # Weight for the current frame's line (decreased from 0.2 for EXTRA smooth)
    MAX_ALLOWED_JUMP_RATIO = 0.25  # Max jump distance as a ratio of image width
    
    # Variables for bubble detection
    bubbles_detected = False
    too_many_bubbles = False
    last_clear_frame_lines = None  # Store the last clear frame's line positions
    last_clear_green_lines = None  # Store the last clear frame's green lines
    bubble_detection_count = 0  # Counter for consecutive bubble detections
    BUBBLE_HYSTERESIS = 2  # Reduced from 3 to 2 for faster response
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add a small delay between frame reads
            time.sleep(0.001)  # 1ms delay
            
            # Display original frame
            cv2.imshow('Original', frame)
            
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur (kernel size 5x5, sigma=1)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)
            # Apply binary thresholding (threshold at 70)
            _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
            
            # Detect bubbles
            current_bubbles_detected, current_too_many_bubbles, circles = detect_bubbles(thresh)
            
            # Update bubble detection state with improved hysteresis
            if current_bubbles_detected:
                bubble_detection_count = min(bubble_detection_count + 1, BUBBLE_HYSTERESIS)  # Slower increase
                bubbles_detected = True
                too_many_bubbles = current_too_many_bubbles
            else:
                bubble_detection_count = max(0, bubble_detection_count - 1)  # Slower decrease
                if bubble_detection_count == 0:
                    bubbles_detected = False
                    too_many_bubbles = False
            
            # Create output image for visualization
            output_lines_on_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Process frame for line detection
            image_height, image_width = thresh.shape[:2]
            image_center_x = image_width / 2.0

            edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=100, maxLineGap=30)

            if lines is not None and not bubbles_detected:  # Only process lines if no bubbles detected
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
                    current_green_lines = []  # Store current green lines
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
                                if not too_many_bubbles:
                                    # Only add to current_green_lines if we have less than 2 lines
                                    if len(current_green_lines) < 2:
                                        current_green_lines.append((valid_pt1, valid_pt2))
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

                        if prev_avg_line_endpoints is not None and not bubbles_detected:
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
                        
                        # Store the lines if no bubbles are detected
                        if not bubbles_detected:
                            last_clear_frame_lines = (display_pt1, display_pt2)
                            last_clear_green_lines = current_green_lines
                            prev_avg_line_endpoints = (current_avg_pt1, current_avg_pt2)
                        
                        # Draw the lines (either current or frozen)
                        if bubbles_detected and last_clear_frame_lines is not None:
                            # When bubbles are detected, use the last clear frame's lines
                            cv2.line(output_lines_on_thresh, last_clear_frame_lines[0], last_clear_frame_lines[1], (255, 0, 0), 8) # Blue
                            # Draw only the two main green lines from the last clear frame
                            if last_clear_green_lines is not None and len(last_clear_green_lines) >= 2:
                                # Sort lines by x-position to ensure consistent left/right ordering
                                sorted_lines = sorted(last_clear_green_lines, key=lambda line: (line[0][0] + line[1][0])/2)
                                # Draw only the leftmost and rightmost lines
                                cv2.line(output_lines_on_thresh, sorted_lines[0][0], sorted_lines[0][1], (0, 255, 0), 8) # Left green
                                cv2.line(output_lines_on_thresh, sorted_lines[-1][0], sorted_lines[-1][1], (0, 255, 0), 8) # Right green
                        else:
                            # When no bubbles or uncertain, use current frame's lines
                            cv2.line(output_lines_on_thresh, display_pt1, display_pt2, (255, 0, 0), 8) # Blue
                            # Draw only the two main green lines from current frame
                            if current_green_lines and len(current_green_lines) >= 2:
                                # Sort lines by x-position to ensure consistent left/right ordering
                                sorted_lines = sorted(current_green_lines, key=lambda line: (line[0][0] + line[1][0])/2)
                                # Draw only the leftmost and rightmost lines
                                cv2.line(output_lines_on_thresh, sorted_lines[0][0], sorted_lines[0][1], (0, 255, 0), 8) # Left green
                                cv2.line(output_lines_on_thresh, sorted_lines[-1][0], sorted_lines[-1][1], (0, 255, 0), 8) # Right green
                    else:
                        # If no two valid lines found, reset smoothing history
                        prev_avg_line_endpoints = None
                        if not bubbles_detected:
                            last_clear_frame_lines = None
                            last_clear_green_lines = None

            # Draw bubble detection status and count
            if bubbles_detected:
                if too_many_bubbles:
                    status_text = f"TOO MANY BUBBLES - LINES FROZEN (Count: {bubble_detection_count})"
                else:
                    status_text = f"BUBBLES DETECTED (Count: {bubble_detection_count})"
                color = (0, 0, 255)  # Red
            else:
                status_text = f"NO BUBBLES DETECTED (Count: {bubble_detection_count})"
                color = (0, 255, 0)  # Green

            # Draw the status text
            cv2.putText(output_lines_on_thresh, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('BW Post Threshold', thresh)
            cv2.imshow('Lines on Thresh', output_lines_on_thresh)

        key = cv2.waitKey(1) & 0xFF  # Changed from 5 to 1 for 4x speed
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 