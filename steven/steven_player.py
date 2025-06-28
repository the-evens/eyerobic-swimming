import cv2
import argparse
import numpy as np
from functions import *
from collections import Counter
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"error: could not open {args.input}")
        return

    target_width = 640
    target_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    paused = False
    SMOOTHING_FACTOR = 0.1 
    prev_center_line = None
    frame_num = 0
    status_history = []  
    last_audio_feedback_time = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % 2 != 0:
                continue

            # forcing size down for videos -----------------------------------------------------------------
            if args.input !=0 :
                current_height, current_width = frame.shape[:2]
                if current_width != target_width or current_height != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            # ----------------------------------------------------------------------------------------------

            thresh = treshold_frame(frame)  
            edges = find_edges(thresh)   
            left, right, center, lines, breathing = find_lines(edges)

            display_frame = frame.copy() 
            cv2.imshow('lines', lines)
            if center:
                if prev_center_line is None:
                    prev_center_line = center
                else:
                    smooth_top_x = int(SMOOTHING_FACTOR * center[0] + (1 - SMOOTHING_FACTOR) * prev_center_line[0])
                    smooth_bottom_x = int(SMOOTHING_FACTOR * center[2] + (1 - SMOOTHING_FACTOR) * prev_center_line[2])
                    smooth_center_line = (smooth_top_x, 0, smooth_bottom_x, display_frame.shape[0])
                    
                    cv2.line(display_frame, (smooth_center_line[0], smooth_center_line[1]), 
                            (smooth_center_line[2], smooth_center_line[3]), (0, 0, 255), 5)
                    prev_center_line = smooth_center_line
                    t_marker_frame, t_marker_lines = find_t_marker(edges)

            elif prev_center_line is not None:
                cv2.line(display_frame, (prev_center_line[0], prev_center_line[1]), 
                        (prev_center_line[2], prev_center_line[3]), (0, 0, 255), 5)

            wall_detected = t_marker_frame is not None if 't_marker_frame' in locals() else False
            current_center_line = center if center else prev_center_line
            current_status = get_current_status(current_center_line, wall_detected, breathing, display_frame)
            
            status_history.append(current_status)
            if len(status_history) > 90:
                status_history.pop(0)
            
            display_frame = display_status_with_history(display_frame, status_history)
            
            if len(status_history) > 0:
                most_common_status = Counter(status_history).most_common(1)[0][0]
                last_audio_feedback_time = play_audio_feedback(most_common_status, last_audio_feedback_time)
                
            cv2.imshow('original', display_frame)

        key = cv2.waitKey(8) & 0xFF
        if key == 27: # esc leaves
            break
        elif key == 32: # space pauses
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 

