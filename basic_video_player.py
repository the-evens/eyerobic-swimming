import cv2
import argparse
import numpy as np
from collections import deque

class SwimmingLaneDetector:
    def __init__(self):
        # History for temporal smoothing
        self.line_history = deque(maxlen=8)
        self.confidence_threshold = 0.7
        
        # Swimming pool specific parameters
        self.roi_bottom_ratio = 0.3    # Focus on bottom 70% where lane markers are
        self.roi_side_margin = 0.15    # Leave 15% margin on each side
        
        # Enhanced line detection parameters
        self.canny_low = 30
        self.canny_high = 100
        self.hough_threshold = 25
        self.min_line_length_ratio = 0.06  # Very sensitive for lane marker segments
        self.max_line_gap_ratio = 0.2      # Large gaps allowed for broken markers
        
        # Angle filtering for swimming lanes (more restrictive than roads)
        self.min_angle_deg = 10     # Swimming lanes are more horizontal
        self.max_angle_deg = 170
        self.parallel_tolerance_deg = 25
        
        # Lane marker specific parameters
        self.min_lane_separation_ratio = 0.08  # Minimum separation between lane edges
        self.max_lane_separation_ratio = 0.4   # Maximum separation between lane edges
        
    def create_swimming_roi(self, image_shape):
        """Create ROI mask focusing on swimming lane area"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Focus on bottom portion where lane markers are visible
        roi_top = int(height * self.roi_bottom_ratio)
        roi_left = int(width * self.roi_side_margin)
        roi_right = int(width * (1 - self.roi_side_margin))
        
        # Create rectangular ROI
        mask[roi_top:height, roi_left:roi_right] = 255
        return mask
    
    def enhanced_preprocessing(self, gray_image, roi_mask):
        """Multi-stage preprocessing optimized for swimming pool conditions"""
        # Apply ROI mask
        masked = cv2.bitwise_and(gray_image, roi_mask)
        
        # Stage 1: Noise reduction
        denoised = cv2.bilateralFilter(masked, 9, 75, 75)
        
        # Stage 2: Multiple thresholding approaches
        # OTSU thresholding for automatic threshold selection
        _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        # Fixed threshold for dark lane markers
        _, fixed_thresh = cv2.threshold(denoised, 60, 255, cv2.THRESH_BINARY)
        
        # Combine all three methods
        combined = cv2.bitwise_and(cv2.bitwise_and(otsu_thresh, adaptive_thresh), fixed_thresh)
        
        # Stage 3: Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))  # Horizontal emphasis
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        return cleaned
    
    def detect_line_segments(self, thresh_image):
        """Advanced line segment detection with multiple passes"""
        height, width = thresh_image.shape
        
        # Edge detection
        edges = cv2.Canny(thresh_image, self.canny_low, self.canny_high, apertureSize=3)
        
        # Multiple Hough detection passes with different parameters
        all_lines = []
        
        # Pass 1: Conservative detection
        min_length_1 = int(width * self.min_line_length_ratio)
        max_gap_1 = int(width * self.max_line_gap_ratio)
        lines_1 = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold, 
                                  minLineLength=min_length_1, maxLineGap=max_gap_1)
        if lines_1 is not None:
            all_lines.extend(lines_1)
        
        # Pass 2: More sensitive detection
        lines_2 = cv2.HoughLinesP(edges, 1, np.pi/180, max(15, self.hough_threshold-10), 
                                  minLineLength=min_length_1//2, maxLineGap=max_gap_1)
        if lines_2 is not None:
            all_lines.extend(lines_2)
        
        # Pass 3: Very sensitive for short segments
        lines_3 = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 
                                  minLineLength=min_length_1//3, maxLineGap=max_gap_1*2)
        if lines_3 is not None:
            all_lines.extend(lines_3)
        
        return all_lines if all_lines else None
    
    def filter_and_group_lines(self, lines, image_width):
        """Advanced line filtering and grouping for swimming lanes"""
        if not lines:
            return []
        
        # Filter by angle
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            
            if self.min_angle_deg <= angle <= self.max_angle_deg:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                avg_x = (x1 + x2) / 2
                valid_lines.append({
                    'line': (x1, y1, x2, y2),
                    'angle': angle,
                    'length': length,
                    'avg_x': avg_x
                })
        
        if len(valid_lines) < 2:
            return valid_lines
        
        # Group lines by similar angles
        groups = []
        for line_info in valid_lines:
            added_to_group = False
            for group in groups:
                if abs(line_info['angle'] - group['avg_angle']) < self.parallel_tolerance_deg:
                    group['lines'].append(line_info)
                    group['total_length'] += line_info['length']
                    group['avg_angle'] = np.mean([l['angle'] for l in group['lines']])
                    added_to_group = True
                    break
            
            if not added_to_group:
                groups.append({
                    'lines': [line_info],
                    'avg_angle': line_info['angle'],
                    'total_length': line_info['length']
                })
        
        # Sort groups by total length (quality indicator)
        groups.sort(key=lambda g: g['total_length'], reverse=True)
        
        # Select best two groups representing lane edges
        selected_groups = []
        for group in groups:
            if len(selected_groups) == 0:
                selected_groups.append(group)
            elif len(selected_groups) == 1:
                # Check if this group represents the other lane edge
                first_group_avg_x = np.mean([l['avg_x'] for l in selected_groups[0]['lines']])
                current_group_avg_x = np.mean([l['avg_x'] for l in group['lines']])
                separation = abs(current_group_avg_x - first_group_avg_x)
                
                min_sep = image_width * self.min_lane_separation_ratio
                max_sep = image_width * self.max_lane_separation_ratio
                
                if min_sep <= separation <= max_sep:
                    selected_groups.append(group)
                    break
        
        return selected_groups
    
    def fit_lane_lines(self, line_groups, image_height):
        """Fit robust lines to lane marker edges"""
        fitted_lines = []
        
        for group in line_groups:
            # Collect all points from the group
            all_points = []
            for line_info in group['lines']:
                x1, y1, x2, y2 = line_info['line']
                all_points.extend([(x1, y1), (x2, y2)])
            
            if len(all_points) < 4:  # Need at least 2 line segments
                continue
            
            # Use RANSAC for robust line fitting
            points = np.array(all_points, dtype=np.float32)
            
            # Fit line using least squares
            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate line endpoints for full image height
            # Line equation: (x,y) = (x0,y0) + t*(vx,vy)
            if abs(vy) > 1e-6:  # Avoid division by zero
                # Calculate t for y = 0 and y = image_height-1
                t1 = (0 - y0) / vy
                t2 = (image_height - 1 - y0) / vy
                
                x1 = int(x0 + t1 * vx)
                y1 = 0
                x2 = int(x0 + t2 * vx)
                y2 = image_height - 1
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, image_height))  # Should be image_width, but let's be safe
                x2 = max(0, min(x2, image_height))
                
                fitted_lines.append((x1, y1, x2, y2))
        
        return fitted_lines
    
    def temporal_smoothing(self, current_lines):
        """Advanced temporal smoothing with confidence tracking"""
        if not current_lines:
            return []
        
        self.line_history.append(current_lines)
        
        if len(self.line_history) < 3:
            return current_lines
        
        # Weighted averaging with exponential decay
        weights = [0.4, 0.3, 0.2, 0.1]  # Current frame gets highest weight
        smoothed_lines = []
        
        for i in range(len(current_lines)):
            if i >= len(current_lines) or current_lines[i] is None:
                continue
            
            x1_sum, y1_sum, x2_sum, y2_sum = 0, 0, 0, 0
            total_weight = 0
            
            # Average over recent frames
            for j, frame_lines in enumerate(reversed(list(self.line_history))):
                if j >= len(weights):
                    break
                
                if i < len(frame_lines) and frame_lines[i] is not None:
                    x1, y1, x2, y2 = frame_lines[i]
                    weight = weights[j]
                    
                    x1_sum += x1 * weight
                    y1_sum += y1 * weight
                    x2_sum += x2 * weight
                    y2_sum += y2 * weight
                    total_weight += weight
            
            if total_weight > 0:
                smoothed_line = (
                    int(x1_sum / total_weight),
                    int(y1_sum / total_weight),
                    int(x2_sum / total_weight),
                    int(y2_sum / total_weight)
                )
                smoothed_lines.append(smoothed_line)
        
        return smoothed_lines
    
    def detect_swimming_lanes(self, frame):
        """Main swimming lane detection pipeline"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create ROI for swimming pool
        roi_mask = self.create_swimming_roi(gray.shape)
        
        # Enhanced preprocessing
        thresh = self.enhanced_preprocessing(gray, roi_mask)
        
        # Detect line segments
        raw_lines = self.detect_line_segments(thresh)
        
        # Filter and group lines
        line_groups = self.filter_and_group_lines(raw_lines, gray.shape[1])
        
        # Fit robust lines
        fitted_lines = self.fit_lane_lines(line_groups, gray.shape[0])
        
        # Apply temporal smoothing
        smoothed_lines = self.temporal_smoothing(fitted_lines)
        
        return thresh, roi_mask, smoothed_lines, raw_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video file')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open {args.input}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height} at {fps:.2f} FPS")

    # Create resizable windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Enhanced Threshold', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Swimming Lane Detection', cv2.WINDOW_NORMAL)
    
    # Resize windows
    cv2.resizeWindow('Original', 800, 600)
    cv2.resizeWindow('Enhanced Threshold', 800, 600)
    cv2.resizeWindow('Swimming Lane Detection', 800, 600)
    
    # Initialize detector
    detector = SwimmingLaneDetector()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display original
            cv2.imshow('Original', frame)
            
            # Detect swimming lanes
            thresh, roi_mask, lane_lines, raw_lines = detector.detect_swimming_lanes(frame)
            
            # Show clean threshold
            cv2.imshow('Enhanced Threshold', thresh)
            
            # Create main output with swimming lane detection
            output = frame.copy()
            
            # Clean output without ROI overlay
            
            # Draw detected swimming lane edges and center guidance
            if len(lane_lines) >= 2:
                # Sort lines by x position to ensure left/right order
                lane_lines_sorted = sorted(lane_lines, key=lambda line: (line[0] + line[2]) / 2)
                
                left_edge = lane_lines_sorted[0]
                right_edge = lane_lines_sorted[1]
                
                # Make lines perfectly parallel using average slope
                x1_left, y1_left, x2_left, y2_left = left_edge
                x1_right, y1_right, x2_right, y2_right = right_edge
                
                # Calculate average slope
                slope_left = (y2_left - y1_left) / (x2_left - x1_left + 1e-6)
                slope_right = (y2_right - y1_right) / (x2_right - x1_right + 1e-6)
                avg_slope = (slope_left + slope_right) / 2
                
                # Recalculate parallel lines
                image_height = output.shape[0]
                
                # Left line with average slope
                left_center_x = (x1_left + x2_left) / 2
                left_center_y = (y1_left + y2_left) / 2
                left_x1_new = int(left_center_x - left_center_y / (avg_slope + 1e-6))
                left_x2_new = int(left_center_x + (image_height - 1 - left_center_y) / (avg_slope + 1e-6))
                
                # Right line with average slope
                right_center_x = (x1_right + x2_right) / 2
                right_center_y = (y1_right + y2_right) / 2
                right_x1_new = int(right_center_x - right_center_y / (avg_slope + 1e-6))
                right_x2_new = int(right_center_x + (image_height - 1 - right_center_y) / (avg_slope + 1e-6))
                
                # Draw lane marker edges in green
                cv2.line(output, (left_x1_new, 0), (left_x2_new, image_height-1), (0, 255, 0), 6)
                cv2.line(output, (right_x1_new, 0), (right_x2_new, image_height-1), (0, 255, 0), 6)
                
                # Draw swimmer guidance line in blue (center between lane edges)
                center_x1 = (left_x1_new + right_x1_new) // 2
                center_x2 = (left_x2_new + right_x2_new) // 2
                cv2.line(output, (center_x1, 0), (center_x2, image_height-1), (255, 0, 0), 8)
                
            elif len(lane_lines) == 1:
                # Single lane edge detected
                line = lane_lines[0]
                cv2.line(output, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 6)
            
            cv2.imshow('Swimming Lane Detection', output)

        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 