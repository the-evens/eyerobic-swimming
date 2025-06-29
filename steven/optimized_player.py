import cv2
import numpy as np
import time
import pygame
from optimized_functions import *
from collections import Counter
    
def main():

    # camera setup ----------------------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error: could not open camera")
        return

    target_width = 640
    target_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # ------------------------------------------------------------------------------------------------------

    # player setup ----------------------------------------------------------------------------------------
    paused = False
    SMOOTHING_FACTOR = 0.1 
    prev_center_line = None
    frame_num = 0
    status_history = []  
    last_audio_feedback_time = 0
    # ------------------------------------------------------------------------------------------------------

    # recording setup -------------------------------------------------------------------------------------
    time_limit = 15 # CHANGE THIS TO CHANGE THE TIME LIMIT
    start_time = time.time()
    recording_started = False
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"recorded_frames_{int(start_time)}.mp4"
    fps = 30
    out = cv2.VideoWriter(output_filename, fourcc, fps, (target_width, target_height))
    recording_started = True
    print(f"Started recording frames to {output_filename}")
    
    # play startup sound ----------------------------------------------------------------------------------
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    startup_beep = generate_beep(duration=0.3, frequency=2000, channel='both', volume=0.8)
    startup_beep.play()
    # ------------------------------------------------------------------------------------------------------

    # main loop ------------------------------------------------------------------------------------------
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % 2 != 0:
                continue
            
            # process frame --------------------------------------------------------------------------------
            thresh = treshold_frame(frame)  
            edges = find_edges(thresh)   
            left, right, center, breathing = find_lines(edges)
            display_frame = frame.copy() 
            # --------------------------------------------------------------------------------------------------

            t_marker_detected = False
            # find center line --------------------------------------------------------------------------------
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
                    t_marker_detected = find_t_marker(edges)

            elif prev_center_line is not None:
                cv2.line(display_frame, (prev_center_line[0], prev_center_line[1]), 
                        (prev_center_line[2], prev_center_line[3]), (0, 0, 255), 5)
            # --------------------------------------------------------------------------------------------------

            # check for wall ----------------------------------------------------------------------------------
            wall_detected = t_marker_detected
            current_center_line = center if center else prev_center_line
            current_status = get_current_status(current_center_line, wall_detected, breathing, display_frame)
            # --------------------------------------------------------------------------------------------------

            # update status history --------------------------------------------------------------------------
            status_history.append(current_status)
            if len(status_history) > 90:
                status_history.pop(0)
            # --------------------------------------------------------------------------------------------------

            # display status ----------------------------------------------------------------------------------
            display_frame = display_status_with_history(display_frame, status_history)
            
            if len(status_history) > 0:
                most_common_status = Counter(status_history).most_common(1)[0][0]
                last_audio_feedback_time = play_audio_feedback(most_common_status, last_audio_feedback_time)
            # --------------------------------------------------------------------------------------------------

            # save frame to recording ------------------------------------------------------------------------
            if recording_started and out is not None:
                out.write(display_frame)
            # --------------------------------------------------------------------------------------------------

            # check if time limit has been reached ----------------------------------------------------------------
            current_time = time.time()
            if current_time - start_time >= time_limit:
                print("15 seconds completed - stopping recording and exiting")
                break
            # --------------------------------------------------------------------------------------------------

    # clean up ---------------------------------------------------------------------------------------------
    if out is not None:
        out.release()
        print(f"Recording saved to {output_filename}")
    # ------------------------------------------------------------------------------------------------------
    
    cap.release()

if __name__ == "__main__":
    main() 

