import cv2
import numpy as np
import pygame
import time
import io
import wave

def treshold_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    return thresh

def find_edges(frame):
    edges = cv2.Canny(frame, 100, 200, apertureSize=3)
    return edges

def find_lines(frame):
    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, threshold=75, minLineLength=75, maxLineGap=75)
    height, width = frame.shape[:2]
    
    if lines is not None:
        lines_reshaped = lines.reshape(-1, 4)

        dx = lines_reshaped[:, 2] - lines_reshaped[:, 0]
        dy = lines_reshaped[:, 3] - lines_reshaped[:, 1]
        
        vertical_mask = (dx == 0) | (np.abs(dy / np.where(dx == 0, 1, dx)) > 2)
        vertical_lines = lines_reshaped[vertical_mask].tolist()
        
        if len(vertical_lines) >= 8: # breathing
            return None, None, None, True

        if len(vertical_lines) >= 2:
            lines_array = np.array(vertical_lines)
            
            x_centers = (lines_array[:, 0] + lines_array[:, 2]) / 2
            leftmost_idx = np.argmin(x_centers)
            rightmost_idx = np.argmax(x_centers)
            
            line1 = vertical_lines[leftmost_idx]
            line2 = vertical_lines[rightmost_idx]
            max_gap = abs(x_centers[rightmost_idx] - x_centers[leftmost_idx])
            
            if line1 and line2:
                def extend_line(x1, y1, x2, y2):
                    if x2 - x1 == 0:
                        return (x1, 0, x1, height)
                    slope = (y2 - y1) / (x2 - x1)
                    return (0, int(y1 - slope * x1), width, int(y1 + slope * (width - x1)))
                
                ext1 = extend_line(*line1)
                ext2 = extend_line(*line2)
                
                left_line = ext1 if ext1[0] < ext2[0] else ext2
                right_line = ext2 if ext1[0] < ext2[0] else ext1
            
                def get_x_at_y(line, y):
                    x1, y1, x2, y2 = line
                    return x1 if x2 - x1 == 0 else int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                
                top_x = (get_x_at_y(left_line, 0) + get_x_at_y(right_line, 0)) // 2
                bottom_x = (get_x_at_y(left_line, height) + get_x_at_y(right_line, height)) // 2
                center_line = (top_x, 0, bottom_x, height)
                
                return left_line, right_line, center_line, False
            
    return None, None, None, False

def find_t_marker(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=75)
    frame_center_x = edges.shape[1] // 2

    if lines is not None:
        center_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1 and abs((y2 - y1) / (x2 - x1)) < 1.5:  # horizontal-ish
                line_center_x = (x1 + x2) / 2
                if abs(line_center_x - frame_center_x) < 100:  # near center
                    center_lines.append((x1, y1, x2, y2))
        
        if center_lines:
            return True
        
    return False


def generate_beep(duration=0.3, frequency=800, sample_rate=44100, channel='both', volume=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = np.sin(frequency * 2 * np.pi * t) * volume

    beep_int = (beep * 32767).astype(np.int16)
    if channel == 'left':
        stereo_beep = np.column_stack([beep_int, np.zeros_like(beep_int)])
    elif channel == 'right':
        stereo_beep = np.column_stack([np.zeros_like(beep_int), beep_int])
    else:  # both channels
        stereo_beep = np.column_stack([beep_int, beep_int])
    
    sound = pygame.sndarray.make_sound(stereo_beep)
    return sound


def play_audio_feedback(status, last_feedback_time):
    current_time = time.time()
    
    if current_time - last_feedback_time < 2.0:
        return last_feedback_time

    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    
    if status == "GO LEFT":
        beep = generate_beep(channel='left')
        beep.play()
        return current_time
        
    elif status == "GO RIGHT":
        beep = generate_beep(channel='right')
        beep.play()
        return current_time
        
    elif status == "WALL AHEAD":
        beep = generate_beep(frequency=1000, channel='both')  # higher pitch for wall warning
        beep.play()
        return current_time
    
    return last_feedback_time

def get_current_status(center_line, wall_detected, breathing, frame):

    if breathing:
        return "BREATHING"
    elif wall_detected:
        return "WALL AHEAD"
    elif center_line is not None:
        center_x = (center_line[0] + center_line[2]) // 2
        frame_center = frame.shape[1] // 2
        offset = center_x - frame_center
        
        if offset > 50:
            return "GO RIGHT"
        elif offset < -50:
            return "GO LEFT"
        else:
            return "CENTERED"
    else:
        return "NONE"


def display_status_with_history(frame, status_history):
    from collections import Counter
    
    if len(status_history) > 0:
        most_common_status = Counter(status_history).most_common(1)[0][0]
    else:
        most_common_status = "NONE"
    
    if most_common_status == "BREATHING":
        color = (0, 255, 0)  # green
    elif most_common_status == "WALL AHEAD":
        color = (0, 0, 255)  # red
    elif most_common_status in ["GO RIGHT", "GO LEFT"]:
        color = (0, 165, 255)  # orange
    elif most_common_status == "CENTERED":
        color = (255, 255, 255)  # white
    else:
        color = (128, 128, 128)  # gray
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    text_size = cv2.getTextSize(most_common_status, font, font_scale, thickness)[0]
    text_x = frame.shape[1] - text_size[0] - 20
    text_y = 40
    
    cv2.rectangle(frame, 
                  (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, most_common_status, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame
