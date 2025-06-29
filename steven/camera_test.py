import libcamera
import time

camera = libcamera.CameraManager().get_camera()
camera.start()

# Let the camera warm up
time.sleep(1)

# Capture a single frame
frame = camera.capture()

# Save it as a JPEG
with open("image.jpg", "wb") as f:
    f.write(frame)

camera.stop()
