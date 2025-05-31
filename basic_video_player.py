import cv2
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basic video player')
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return
    
    # Create window
    window_name = "Video Player"
    cv2.namedWindow(window_name)
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        # Wait for 30ms for a key press (roughly 30fps when not paused)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to pause/resume
            paused = not paused
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 