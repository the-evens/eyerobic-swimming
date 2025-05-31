import cv2
import numpy as np
import math
import collections
import os
import argparse
import time

# --- Core Tracking Functions ---

def calculate_deviation(center_x, frame_width):
    """Calculate horizontal deviation from lane center in pixels"""
    return center_x - (frame_width // 2)

def process_frame(frame, lower_color, upper_color, frame_count):
    """Process a single frame to detect and track the T-marker
    Returns: (marker_detected, center_x, center_y, deviation)"""
    frame_height, frame_width = frame.shape[:2]
    
    # Detect T-marker using color thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Clean up mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find T-marker contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, frame_width // 2, frame_height // 2, 0
        
    # Get largest contour (likely the T-marker)
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) <= 100:
        return False, frame_width // 2, frame_height // 2, 0
        
    # Calculate T-marker center
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return False, frame_width // 2, frame_height // 2, 0
        
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    deviation = calculate_deviation(center_x, frame_width)
    
    # Draw marker visualization
    draw_marker_highlight(frame, largest_contour, center_x, center_y, frame_count)
    
    return True, center_x, center_y, deviation

# --- Visualization Functions ---

def draw_marker_highlight(frame, contour, center_x, center_y, frame_count):
    """Draw visual elements highlighting the detected T-marker"""
    frame_height, frame_width = frame.shape[:2]
    
    # Draw contour
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
    
    # Draw glowing rectangle
    x, y, w, h = cv2.boundingRect(contour)
    padding = 10
    highlight = frame.copy()
    cv2.rectangle(highlight, 
                 (max(0, x-padding), max(0, y-padding)), 
                 (min(frame_width, x+w+padding), min(frame_height, y+h+padding)), 
                 (0, 255, 255), -1)
    cv2.addWeighted(highlight, 0.3, frame, 0.7, 0, frame)
    
    # Draw pulsing center ring
    ring_radius = 15 + int(5 * np.sin(frame_count * 0.2))
    cv2.circle(frame, (center_x, center_y), ring_radius, (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), ring_radius - 5, (255, 255, 0), 1)
    
    # Draw vertical guide line
    for i in range(center_y, frame_height, 3):
        intensity = 255 - int(((i - center_y) / (frame_height - center_y)) * 155)
        thickness = 2 if i % 15 < 8 else 1  # Dashed line effect
        cv2.line(frame, (center_x, i), (center_x, i+2), (0, intensity, 255), thickness)

def draw_metrics_overlay(frame, marker_detected, smoothed_deviation, center_x):
    """Draw metrics overlay in top right corner"""
    frame_height, frame_width = frame.shape[:2]
    overlay_width, overlay_height = 250, 150
    
    # Create overlay background with gradient
    overlay_bg = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
    for i in range(overlay_height):
        cv2.line(overlay_bg, (0, i), (overlay_width, i), (40, 40, 50), 1)
    
    # Add metrics
    metrics = [
        ("SWIM METRICS", None, 20, 0.6, (180, 180, 180)),
        (f"Marker Detected: {'Yes' if marker_detected else 'No'}", 50, 0.5, (0, 255, 255)),
        (f"Deviation: {smoothed_deviation:.1f}px", 75, 0.5, (0, 255, 255)),
        (f"Position: {center_x}px", 100, 0.5, (0, 255, 255)),
        (f"Dev %: {(smoothed_deviation / (frame_width/2) * 100):.1f}%", 125, 0.5, (0, 255, 255))
    ]
    
    for metric in metrics:
        if len(metric) == 5:  # Title
            text, y_pos, size, color = metric[0], metric[2], metric[3], metric[4]
        else:  # Regular metric
            text, y_pos, size, color = metric
            cv2.rectangle(overlay_bg, (5, y_pos-15), (overlay_width-5, y_pos+5), (60, 60, 60), -1)
        cv2.putText(overlay_bg, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    
    # Add overlay to frame
    metrics_x = frame_width - overlay_width - 80
    metrics_y = frame_height // 4 - overlay_height // 2
    frame[metrics_y:metrics_y+overlay_height, metrics_x:metrics_x+overlay_width] = overlay_bg

def draw_deviation_graph(frame, deviation_history, max_history=100):
    """Draw deviation history graph in bottom right corner"""
    graph_width, graph_height = 300, 150
    graph_bg = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    
    # Draw background and grid
    cv2.rectangle(graph_bg, (0, 0), (graph_width, graph_height), (40, 40, 40), -1)
    center_y = graph_height // 2
    cv2.line(graph_bg, (0, center_y), (graph_width, center_y), (100, 100, 100), 1)
    
    # Plot deviation line
    if deviation_history:
        max_dev = max(50, max([abs(d) for d in deviation_history]))
        scale = (graph_height // 2) / max_dev
        points = []
        
        for i, dev in enumerate(deviation_history):
            x = graph_width - (len(deviation_history) - i) * graph_width // max_history
            y = center_y - int(dev * scale)
            points.append((x, y))
            
        for i in range(1, len(points)):
            color = (0, 0, 255) if deviation_history[i] > 0 else (255, 0, 0)
            cv2.line(graph_bg, points[i-1], points[i], color, 2)
    
    # Add current value
    if deviation_history:
        cv2.putText(graph_bg, f"{deviation_history[-1]:.1f}px", 
                    (graph_width - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Add graph to frame
    graph_y = 3 * frame.shape[0] // 4 - graph_height // 2
    graph_x = frame.shape[1] - graph_width - 80
    frame[graph_y:graph_y+graph_height, graph_x:graph_x+graph_width] = graph_bg

# --- Color Calibration ---

def run_calibration(video_path):
    """Run interactive color calibration tool"""
    # Create calibration window and trackbars
    cv2.namedWindow('Color Calibration')
    for name, default in [
        ('H_Lower', 90), ('S_Lower', 50), ('V_Lower', 50),
        ('H_Upper', 130), ('S_Upper', 255), ('V_Upper', 255)
    ]:
        cv2.createTrackbar(name, 'Color Calibration', default, 
                          179 if name.startswith('H') else 255, lambda x: None)
    
    # Setup video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        return None, None
    
    print("\nColor Calibration Instructions:")
    print("1. Use sliders to adjust HSV range")
    print("2. Goal: Highlight only the T-marker")
    print("3. 'p': Play/pause video")
    print("4. 'c': Save and exit")
    print("5. ESC: Exit without saving\n")
    
    paused = True
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # Get current HSV ranges
        hsv_values = {}
        for name in ['H_Lower', 'S_Lower', 'V_Lower', 'H_Upper', 'S_Upper', 'V_Upper']:
            hsv_values[name] = cv2.getTrackbarPos(name, 'Color Calibration')
        
        # Show calibration preview
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([hsv_values['H_Lower'], hsv_values['S_Lower'], hsv_values['V_Lower']])
        upper_color = np.array([hsv_values['H_Upper'], hsv_values['S_Upper'], hsv_values['V_Upper']])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Show windows
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('c'):
            # Save calibration
            with open('color_config.py', 'w') as f:
                f.write("# T-marker color calibration\n")
                f.write(f"LOWER_COLOR = [{hsv_values['H_Lower']}, {hsv_values['S_Lower']}, {hsv_values['V_Lower']}]\n")
                f.write(f"UPPER_COLOR = [{hsv_values['H_Upper']}, {hsv_values['S_Upper']}, {hsv_values['V_Upper']}]\n")
            print("Calibration saved to color_config.py")
            return lower_color, upper_color
    
    cap.release()
    cv2.destroyAllWindows()
    return None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time swimming pool T-marker tracking')
    parser.add_argument('--input', default='swimming.mp4', help='Input video file')
    parser.add_argument('--calibrate', action='store_true', help='Run color calibration')
    args = parser.parse_args()
    
    # Setup color ranges
    lower_color = np.array([90, 50, 50])
    upper_color = np.array([130, 255, 255])
    
    if args.calibrate:
        print("Running color calibration...")
        cal_lower, cal_upper = run_calibration(args.input)
        if cal_lower is not None:
            lower_color, upper_color = cal_lower, cal_upper
    else:
        try:
            from . import color_config
            lower_color = np.array(color_config.LOWER_COLOR)
            upper_color = np.array(color_config.UPPER_COLOR)
            print("Using saved calibration settings")
        except ImportError:
            print("Using default color settings")
    
    # Initialize video processing
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create display window
    window_name = "T-Marker Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    # Initialize tracking variables
    deviation_history = collections.deque(maxlen=100)
    smoothed_deviation = 0
    alpha = 0.3  # Smoothing factor
    frame_count = 0
    
    print("\nControls:")
    print("ESC: Exit")
    print("SPACE: Pause/Resume")
    print("R: Reset video to beginning\n")
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            
            # Draw lane center line
            cv2.line(frame, (frame_width//2, 0), (frame_width//2, frame_height), 
                     (0, 0, 255), 1)
            
            # Process frame
            marker_detected, center_x, center_y, deviation = process_frame(
                frame, lower_color, upper_color, frame_count)
            
            # Update smoothed deviation
            if marker_detected:
                smoothed_deviation = (alpha * deviation + 
                                    (1 - alpha) * smoothed_deviation)
                deviation_history.append(smoothed_deviation)
            
            # Draw visualizations
            draw_metrics_overlay(frame, marker_detected, smoothed_deviation, center_x)
            draw_deviation_graph(frame, list(deviation_history))
            
            # Display frame
            cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif key == ord('r'):  # R key
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            deviation_history.clear()
            smoothed_deviation = 0
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 