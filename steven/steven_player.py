import cv2
import argparse
import numpy as np
from collections import deque

def treshold_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1) # tentatively 5x5, sigma 1
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY) # need to tune
    return thresh

def find_edges(frame):
    edges = cv2.Canny(frame, 100, 200, apertureSize=3) # need to tune
    return edges

def find_lines(frame):
    # detect lines
    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, threshold=75, minLineLength=300, maxLineGap=250) # good tuning can basically be perfect
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    
    if lines is not None:
        # filter for vertical lines only
        vertical_lines = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in [line[0]] 
                         if x2 - x1 == 0 or abs((y2 - y1) / (x2 - x1)) > 2]
        
        if len(vertical_lines) >= 2:
            # find pair of lines with largest horizontal gap
            max_gap = 0
            line1 = line2 = None
            for i in range(len(vertical_lines)):
                for j in range(i + 1, len(vertical_lines)):
                    x1 = (vertical_lines[i][0] + vertical_lines[i][2]) // 2
                    x2 = (vertical_lines[j][0] + vertical_lines[j][2]) // 2
                    gap = abs(x2 - x1)
                    if gap > max_gap:
                        max_gap = gap
                        line1, line2 = vertical_lines[i], vertical_lines[j]
            
            # finding average (vibe coded) ------------------------------------------------------------------------
            if line1 and line2:
                # extend lines to frame edges while preserving angle
                def extend_line(x1, y1, x2, y2):
                    if x2 - x1 == 0:
                        return (x1, 0, x1, height)
                    slope = (y2 - y1) / (x2 - x1)
                    return (0, int(y1 - slope * x1), width, int(y1 + slope * (width - x1)))
                
                ext1 = extend_line(*line1)
                ext2 = extend_line(*line2)
                
                # determine left and right lines
                left_line = ext1 if ext1[0] < ext2[0] else ext2
                right_line = ext2 if ext1[0] < ext2[0] else ext1
                
                # draw lines
                cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 10)
                cv2.line(frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)
                
                # calculate center line
                def get_x_at_y(line, y):
                    x1, y1, x2, y2 = line
                    return x1 if x2 - x1 == 0 else int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                
                top_x = (get_x_at_y(left_line, 0) + get_x_at_y(right_line, 0)) // 2
                bottom_x = (get_x_at_y(left_line, height) + get_x_at_y(right_line, height)) // 2
                center_line = (top_x, 0, bottom_x, height)
                cv2.line(frame, (center_line[0], center_line[1]), (center_line[2], center_line[3]), (0, 0, 255), 5)
                
                return left_line, right_line, center_line, frame
            # ----------------------------------------------------------------------------------------
            
    return None

def find_t_marker(edges, left_line, right_line):
    """Detect T-marker by finding horizontal lines between the lane lines"""
    if left_line is None or right_line is None:
        return False, None, None, None
    
    # Detect horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal (small slope)
            if abs(y2 - y1) < 20 and abs(x2 - x1) > 50:
                # Get x-coordinates of lane lines at this y-position
                y_avg = (y1 + y2) // 2
                
                # Calculate x positions of lane lines at y_avg
                def get_x_at_y(line, y):
                    x1, y1, x2, y2 = line
                    if y2 - y1 == 0:
                        return x1
                    return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                
                left_x = get_x_at_y(left_line, y_avg)
                right_x = get_x_at_y(right_line, y_avg)
                
                # Check if horizontal line is between lane lines
                line_left = min(x1, x2)
                line_right = max(x1, x2)
                
                if line_left < right_x and line_right > left_x:
                    # Check if in upper portion of frame (approaching)
                    if y_avg < edges.shape[0] * 0.6:
                        return True, y_avg, left_x, right_x
    
    return False, None, None, None

def draw_position_indicator(frame, center_x):
    frame_center = frame.shape[1] // 2
    position = "CENTERED"
    if center_x < frame_center - 200:
        position = "LEFT"
    elif center_x > frame_center + 200:
        position = "RIGHT"
    
    # display text
    text_size = cv2.getTextSize(position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = frame.shape[1] - text_size[0] - 30
    text_y = 60
    
    cv2.rectangle(frame, 
                (text_x - 10, text_y - text_size[1] - 10),
                (text_x + text_size[0] + 10, text_y + 10),
                (0, 0, 0), -1)
    cv2.putText(frame, position, (text_x, text_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

class PositionTracker:
    def __init__(self, frame_interval=5):
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.positions = []
    
    def update(self, center_x, frame_width):
        self.frame_count += 1
        if self.frame_count % self.frame_interval == 0:
            frame_center = frame_width // 2
            position = "CENTERED"
            if center_x < frame_center - 200:
                position = "LEFT"
            elif center_x > frame_center + 200:
                position = "RIGHT"
            self.positions.append((self.frame_count, position, center_x))
    
    def get_positions(self):
        return self.positions

# main function
def main():
    # parse input video file
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()

    # check video file
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"error: could not open {args.input}")
        return

    # setup
    paused = False
    SMOOTHING_FACTOR = 0.1  # smoothing
    prev_center_line = None
    position_tracker = PositionTracker(frame_interval=5)

    # play through video
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # display original frame
            original_frame = frame.copy()
            cv2.imshow('original', frame)

            # threshold frame
            thresh = treshold_frame(frame)
            cv2.imshow('bw post threshold', thresh)

            # find edges
            edges = find_edges(thresh)
            cv2.imshow('edges', edges)

            # find lines
            lines = find_lines(edges)

            # find t-marker
            t_marker_detected, y_avg, left_x, right_x = find_t_marker(edges, lines[0], lines[1]) if lines else (False, None, None, None)

            # smoothing (vibe coded) ------------------------------------------------------------
            if lines:
                left_line, right_line, center_line, frame = lines
                cv2.imshow('lines', frame)
                
                # apply exponential smoothing to center line
                if prev_center_line is None:
                    prev_center_line = center_line
                else:
                    # smooth the top and bottom x-coordinates
                    smooth_top_x = int(SMOOTHING_FACTOR * center_line[0] + (1 - SMOOTHING_FACTOR) * prev_center_line[0])
                    smooth_bottom_x = int(SMOOTHING_FACTOR * center_line[2] + (1 - SMOOTHING_FACTOR) * prev_center_line[2])
                    smooth_center_line = (smooth_top_x, 0, smooth_bottom_x, original_frame.shape[0])
                    
                    # draw smoothed center line on original frame
                    cv2.line(original_frame, (smooth_center_line[0], smooth_center_line[1]), 
                            (smooth_center_line[2], smooth_center_line[3]), (0, 0, 255), 5)
                    
                    # draw position indicator
                    center_x = (smooth_center_line[0] + smooth_center_line[2]) // 2
                    draw_position_indicator(original_frame, center_x)
                    
                    # Update position tracker
                    position_tracker.update(center_x, original_frame.shape[1])
                    
                    # draw wall approaching warning
                    if t_marker_detected:
                        cv2.line(original_frame, (0, y_avg), (frame.shape[1]-1, y_avg), (0, 0, 255), 4)
                        cv2.putText(original_frame, "WALL APPROACHING", (50, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    cv2.imshow('original', original_frame)
                    
                    prev_center_line = smooth_center_line
            elif prev_center_line is not None:
                # if no lines detected, use previous smoothed line
                cv2.line(original_frame, (prev_center_line[0], prev_center_line[1]), 
                        (prev_center_line[2], prev_center_line[3]), (0, 0, 255), 5)
                
                # draw position indicator for previous line
                center_x = (prev_center_line[0] + prev_center_line[2]) // 2
                draw_position_indicator(original_frame, center_x)
                
                # draw wall approaching warning if detected
                if t_marker_detected:
                    cv2.line(original_frame, (0, y_avg), (frame.shape[1]-1, y_avg), (255, 0, 0), 4)
                    cv2.putText(original_frame, "WALL APPROACHING", (50, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow('original', original_frame)
            # ----------------------------------------------------------------------------------

        key = cv2.waitKey(8) & 0xFF # set tentatively to 5
        if key == 27: # esc leaves
            break
        elif key == 32: # space pauses
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print tracked positions at the end
    print("\nTracked Position History (every 5 frames):")
    for frame_num, position, center_x in position_tracker.get_positions():
        print(f"Frame {frame_num}: {position} (center_x: {center_x})")

# program
if __name__ == "__main__":
    main() 

