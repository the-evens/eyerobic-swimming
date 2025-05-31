# Swimming Pool T-Marker Tracker

Tracks T-markers in swimming pool lanes and provides real-time metrics for swim analysis.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default video
python pool_marker_tracker.py

# Run with custom video and calibration
python pool_marker_tracker.py --input your_video.mp4 --calibrate
```

## Features
- T-marker detection and tracking
- Lane deviation measurements
- Real-time metrics overlay
- Deviation history graph
- Wall approach warning
- Color calibration tool

## How It Works

For each video frame, the program:

1. **Color Detection**
   - Converts frame to HSV color space
   - Applies color thresholding to isolate the T-marker
   - Uses calibrated color values (or defaults if not calibrated)

2. **Marker Processing**
   - Cleans up the detected mask using morphological operations
   - Finds the largest contour (assumed to be the T-marker)
   - Calculates the marker's center point

3. **Metrics Calculation**
   - Measures deviation from lane center
   - Smooths measurements to reduce noise
   - Updates deviation history

4. **Visualization**
   - Highlights detected T-marker with glowing effect
   - Shows vertical guide line from marker
   - Displays real-time metrics overlay
   - Draws deviation history graph
   - Shows wall warning in final 3 seconds

## Controls (Calibration Mode)
- Sliders: Adjust HSV color range
- `p`: Play/pause video
- `c`: Save calibration
- `ESC`: Exit without saving

## Requirements

- Python 3.6+
- OpenCV
- NumPy

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your swimming video file (named 'swimming.mov') in the same directory as the script
2. Run the script:

```bash
python pool_marker_tracker.py
```

3. The processed video will be saved as 'swimming_tracked.mp4' in the same directory

### Advanced Usage

The program supports several command-line arguments for more flexibility:

```bash
python pool_marker_tracker.py --input your_video.mov --output result.mp4 --calibrate
```

Arguments:
- `--input`: Specify the input video file (default: swimming.mov)
- `--output`: Specify the output video file (default: swimming_tracked.mp4)
- `--calibrate`: Run the color calibration tool before processing

### Color Calibration

For optimal T-marker detection, you can calibrate the color detection to your specific pool conditions:

1. Run the calibration tool:
```bash
python color_calibration.py --video swimming.mov
```

2. Use the sliders to adjust the HSV values until the T-marker is clearly highlighted in the mask window
3. Press 'p' to play/pause the video
4. Press 'c' to capture the current settings (saves to color_config.py)
5. Press ESC to exit

The main tracking script will automatically use your calibrated settings if available.

## Metrics Displayed

- **Marker Detected**: Indicates whether the T-marker is currently visible
- **Deviation**: Distance in pixels from the center of the lane (smoothed)
- **Position**: X-coordinate of the T-marker in the frame
- **Dev %**: Percentage deviation from the center (normalized)

## Deviation Graph

The bottom right corner of the video shows a real-time graph of the T-marker's deviation from the center lane:

- The horizontal center line represents zero deviation (perfectly centered)
- Red lines indicate deviation to the right
- Blue lines indicate deviation to the left
- The graph shows the historical trend of the swimmer's position relative to the lane center

This visualization helps identify patterns in swimming technique and can be used for stroke correction. 