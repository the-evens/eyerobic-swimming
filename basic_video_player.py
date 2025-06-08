import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '1000000'  # Set to a very high value

import cv2
import argparse
import numpy as np
import time  # Add time module for delay
import queue
import threading

# Add these globals at the top of the file (if not already present)
prev_avg_line_endpoints = None  # For temporal smoothing
SMOOTHING_FACTOR = 0.1          # Weight for the current frame's line
MAX_ALLOWED_JUMP_RATIO = 0.25   # Max jump distance as a ratio of image width
bubble_detected_counter = 0
bubble_clear_counter = 0
BUBBLE_DEBOUNCE_FRAMES = 5  # Number of consecutive frames to confirm bubble/clear

def detect_bubbles(thresh, min_radius=5, max_radius=15, min_circles=5, max_circles=15):
    """Detect bubbles in the thresholded image using Hough Circle Transform"""
    try:
        # Apply additional preprocessing to enhance bubble detection
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=3,
            param1=15,
            param2=15,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        if circles is not None:
            num_circles = len(circles[0])
            confidence = 0
            for circle in circles[0]:
                x, y, r = circle
                if r > min_radius:  # Only count larger circles
                    confidence += 1
            has_bubbles = num_circles >= min_circles and confidence >= min_circles
            too_many_bubbles = num_circles > max_circles
            return has_bubbles, too_many_bubbles, circles[0]
        return False, False, None
    except Exception as e:
        print(f"Bubble detection error: {e}")
        return False, False, None

def process_frame(frame):
    global prev_avg_line_endpoints, bubble_detected_counter, bubble_clear_counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Automatic threshold calculation
    # Method 1: Otsu's method for global threshold
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Sample the center region where lines are expected
    h, w = blurred.shape
    center_region = blurred[0:h//2, w//3:2*w//3]  # Upper 50% height, center third width
    
    # Calculate mean and std of center region
    mean_val = np.mean(center_region)
    std_val = np.std(center_region)
    
    # Adaptive threshold based on statistics
    # Lines are typically darker than background in pools
    adaptive_thresh = mean_val - 1.5 * std_val
    
    # Combine both methods with a weighted average
    final_thresh = 0.6 * adaptive_thresh + 0.4 * otsu_thresh #adjust weights if needed
    final_thresh = np.clip(final_thresh, 30, 200)  # Reasonable bounds
    
    _, thresh = cv2.threshold(blurred, int(final_thresh), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel)

    # Robust bubble detection with debounce
    try:
        has_bubbles, too_many, _ = detect_bubbles(thresh_clean)
    except Exception as e:
        print(f"Bubble detection error: {e}")
        has_bubbles = False

    # Debounce logic
    if has_bubbles:
        bubble_detected_counter += 1
        bubble_clear_counter = 0
    else:
        bubble_clear_counter += 1
        bubble_detected_counter = 0
    debounced_bubbles = bubble_detected_counter >= BUBBLE_DEBOUNCE_FRAMES

    overlay = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)
    image_height, image_width = thresh_clean.shape[:2]
    image_center_x = image_width / 2.0

    # --- Black area detection ---
    black_pixel_count = np.sum(thresh_clean == 0)
    total_pixel_count = thresh_clean.size
    black_ratio = black_pixel_count / total_pixel_count
    BLACK_RATIO_THRESHOLD = 0.30  # 30% black pixels means unclear
    too_much_black = black_ratio > BLACK_RATIO_THRESHOLD

    # --- Line detection only if not too much black and no bubbles ---
    confident = not debounced_bubbles and not too_much_black
    found_confident_lines = False
    if confident:
        try:
            edges = cv2.Canny(thresh_clean, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=100, maxLineGap=30)
            line_candidates = []
            if lines is not None:
                for line_segment in lines:
                    x1, y1, x2, y2 = line_segment[0]
                    avg_x_segment = (x1 + x2) / 2.0
                    distance_to_center = abs(avg_x_segment - image_center_x)
                    angle_rad = np.arctan2(y2 - y1, x2 - x1)
                    angle_deg = np.degrees(angle_rad)
                    # Only consider lines that are sufficiently vertical and close to center
                    if abs(angle_deg) > 45 and distance_to_center < image_width * 0.25:
                        line_candidates.append({'segment': (x1, y1, x2, y2), 'distance': distance_to_center, 'avg_x': avg_x_segment, 'angle': angle_rad})
            selected_lines_data = []
            if line_candidates:
                found_confident_lines = True
                line_candidates.sort(key=lambda item: item['distance'])
                selected_lines_data.append(line_candidates[0])
                if len(selected_lines_data) == 1 and len(line_candidates) > 1:
                    line1_data = selected_lines_data[0]
                    angle_similarity_threshold_rad = 0.1
                    min_avg_x_sep_if_angles_similar = image_width * 0.20
                    min_avg_x_sep_if_angles_different = image_width * 0.05
                    for candidate_data in line_candidates[1:]:
                        norm_angle1 = np.mod(line1_data['angle'], np.pi)
                        norm_angle_candidate = np.mod(candidate_data['angle'], np.pi)
                        angle_diff_rad = abs(norm_angle1 - norm_angle_candidate)
                        min_acute_angle_diff = min(angle_diff_rad, np.pi - angle_diff_rad)
                        required_x_separation = min_avg_x_sep_if_angles_similar
                        if min_acute_angle_diff >= angle_similarity_threshold_rad:
                            required_x_separation = min_avg_x_sep_if_angles_different
                        avg_x_dist = abs(candidate_data['avg_x'] - line1_data['avg_x'])
                        if avg_x_dist >= required_x_separation:
                            selected_lines_data.append(candidate_data)
                            break
            extended_lines_endpoints = []
            for line_data in selected_lines_data:
                x1, y1, x2, y2 = line_data['segment']
                pt1_ext, pt2_ext = None, None
                if abs(x1 - x2) < 1e-6:
                    pt1_ext = (x1, 0)
                    pt2_ext = (x1, image_height - 1)
                elif abs(y1 - y2) < 1e-6:
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)
                    extended_lines_endpoints.append(None)
                    continue
                else:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    intercept_b = y1 - slope * x1
                    x_at_top = (0 - intercept_b) / (slope + 1e-6)
                    x_at_bottom = (image_height - 1 - intercept_b) / (slope + 1e-6)
                    pt1_ext = (int(round(x_at_top)), 0)
                    pt2_ext = (int(round(x_at_bottom)), image_height - 1)
                if pt1_ext and pt2_ext:
                    valid_pt1_x = max(0, min(pt1_ext[0], image_width - 1))
                    valid_pt1_y = max(0, min(pt1_ext[1], image_height - 1))
                    valid_pt2_x = max(0, min(pt2_ext[0], image_width - 1))
                    valid_pt2_y = max(0, min(pt2_ext[1], image_height - 1))
                    valid_pt1 = (valid_pt1_x, valid_pt1_y)
                    valid_pt2 = (valid_pt2_x, valid_pt2_y)
                    if np.sqrt((valid_pt2[0]-valid_pt1[0])**2 + (valid_pt2[1]-valid_pt1[1])**2) > 10:
                        cv2.line(overlay, valid_pt1, valid_pt2, (0, 255, 0), 8)
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
                avg_pt1_y = (l1_pt1[1] + l2_pt1[1]) / 2.0
                avg_pt2_x = (l1_pt2[0] + l2_pt2[0]) / 2.0
                avg_pt2_y = (l1_pt2[1] + l2_pt2[1]) / 2.0
                current_avg_pt1 = (int(round(avg_pt1_x)), int(round(avg_pt1_y)))
                current_avg_pt2 = (int(round(avg_pt2_x)), int(round(avg_pt2_y)))
                display_pt1, display_pt2 = current_avg_pt1, current_avg_pt2
                if prev_avg_line_endpoints is not None:
                    prev_pt1, prev_pt2 = prev_avg_line_endpoints
                    current_mid_x = (current_avg_pt1[0] + current_avg_pt2[0]) / 2.0
                    current_mid_y = (current_avg_pt1[1] + current_avg_pt2[1]) / 2.0
                    prev_mid_x = (prev_pt1[0] + prev_pt2[0]) / 2.0
                    prev_mid_y = (prev_pt1[1] + prev_pt2[1]) / 2.0
                    jump_distance = np.sqrt((current_mid_x - prev_mid_x)**2 + (current_mid_y - prev_mid_y)**2)
                    max_jump = image_width * MAX_ALLOWED_JUMP_RATIO
                    if jump_distance < max_jump:
                        smooth_pt1_x = SMOOTHING_FACTOR * current_avg_pt1[0] + (1 - SMOOTHING_FACTOR) * prev_pt1[0]
                        smooth_pt1_y = SMOOTHING_FACTOR * current_avg_pt1[1] + (1 - SMOOTHING_FACTOR) * prev_pt1[1]
                        smooth_pt2_x = SMOOTHING_FACTOR * current_avg_pt2[0] + (1 - SMOOTHING_FACTOR) * prev_pt2[0]
                        smooth_pt2_y = SMOOTHING_FACTOR * current_avg_pt2[1] + (1 - SMOOTHING_FACTOR) * prev_pt2[1]
                        display_pt1 = (int(round(smooth_pt1_x)), int(round(smooth_pt1_y)))
                        display_pt2 = (int(round(smooth_pt2_x)), int(round(smooth_pt2_y)))
                cv2.line(overlay, display_pt1, display_pt2, (255, 0, 0), 8)  # Blue
                prev_avg_line_endpoints = (current_avg_pt1, current_avg_pt2)
            else:
                found_confident_lines = False
                prev_avg_line_endpoints = None
        except Exception as e:
            print(f"Line detection error: {e}")
            found_confident_lines = False
    # If not confident or no confident lines, just redraw the last confident lines (if available)
    if not (confident and found_confident_lines):
        if prev_avg_line_endpoints is not None:
            prev_pt1, prev_pt2 = prev_avg_line_endpoints
            cv2.line(overlay, prev_pt1, prev_pt2, (255, 0, 0), 8)  # Blue
    # Status text
    if not (confident and found_confident_lines):
        status = "UNCLEAR POSITION"
        color = (0, 0, 255)
    else:
        status = "NO BUBBLES"
        color = (0, 255, 0)
    cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add threshold value display
    cv2.putText(overlay, f"Thresh: {int(final_thresh)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame, thresh_clean, overlay

def worker():
    """Background thread for processing frames"""
    while True:
        frame = proc_q.get()
        if frame is None:
            break
        processed = process_frame(frame)
        if not result_q.full():
            result_q.put(processed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()

    # Open video capture
    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('BW Post Threshold', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Lines on Thresh', cv2.WINDOW_NORMAL)
    
    # Set window properties
    cv2.setWindowProperty('Original', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty('BW Post Threshold', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty('Lines on Thresh', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # Initialize queues and start worker thread
    global proc_q, result_q
    proc_q = queue.Queue(maxsize=8)
    result_q = queue.Queue(maxsize=8)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    paused = False
    fps = 0
    count = 0
    t0 = time.time()
    speed_multiplier = 4.0  # Play at 4x speed
    target_fps = 30.0  # Target FPS for all windows
    frame_interval = 1.0 / target_fps  # Time between frames
    next_frame_time = time.time()

    print("Controls: ESC to quit, SPACE to pause/resume")
    print(f"Playing at {speed_multiplier}x speed")
    print(f"Target FPS: {target_fps}")

    while True:
        current_time = time.time()
        
        if not paused and current_time >= next_frame_time:
            ret, frame = cap.read()
            if not ret:
                break

            # Use the original frame for processing
            if proc_q.empty():
                proc_q.put(frame.copy())

            # Display processed frames if available
            if not result_q.empty():
                original, thresh, overlay = result_q.get()
                cv2.imshow('Original', original)
                cv2.imshow('BW Post Threshold', thresh)
                cv2.imshow('Lines on Thresh', overlay)

            # Update FPS counter
            count += 1
            if current_time - t0 >= 1.0:
                fps = count / (current_time - t0)
                count = 0
                t0 = current_time
                cv2.putText(original, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calculate next frame time
            next_frame_time = current_time + frame_interval

        # Handle key presses with minimal wait time
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
            if not paused:
                next_frame_time = time.time()  # Reset timing when resuming

    # Cleanup
    proc_q.put(None)  # Signal worker to exit
    t.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 