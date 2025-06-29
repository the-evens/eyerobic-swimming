import time
import cv2
from picamera2 import Picamera2

def main():
    target_width = 640
    target_height = 480

    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (target_width, target_height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1)  # Warm-up

    paused = False
    SMOOTHING_FACTOR = 0.1 
    prev_center_line = None
    frame_num = 0
    status_history = []  
    last_audio_feedback_time = 0

    # Video recording setup
    start_time = time.time()
    recording_started = False
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"recorded_frames_{int(start_time)}.mp4"

    fps = 30  # PiCamera2 doesn't expose true FPS easily; set what you want
    out = cv2.VideoWriter(output_filename, fourcc, fps, (target_width, target_height))
    recording_started = True
    print(f"Started recording frames to {output_filename}")

    while True:
        if not paused:
            frame = picam2.capture_array()
            frame_num += 1

            # Your frame processing logic here

            out.write(frame)
            cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    out.release()
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
