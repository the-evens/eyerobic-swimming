import pygame
import numpy as np
import time

def generate_beep(duration=0.3, frequency=800, sample_rate=44100, channel='both', volume=1.0):
    """Generate a beep sound for the specified channel(s)"""
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

def main():
    print("Swimming Audio Feedback Test")
    print("=" * 40)
    
    # Initialize pygame mixer
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    
    print("Testing different audio feedback sounds...")
    print("Make sure you're wearing headphones to test left/right channels!")
    print()
    
    # Test startup sound
    print("1. Playing startup sound (both channels)...")
    startup_beep = generate_beep(duration=0.3, frequency=2000, channel='both', volume=0.8)
    startup_beep.play()
    time.sleep(1.5)
    
    # Test left channel (GO LEFT feedback)
    print("2. Playing GO LEFT feedback (left channel only)...")
    left_beep = generate_beep(channel='left')
    left_beep.play()
    time.sleep(1.5)
    
    # Test right channel (GO RIGHT feedback)
    print("3. Playing GO RIGHT feedback (right channel only)...")
    right_beep = generate_beep(channel='right')
    right_beep.play()
    time.sleep(1.5)
    
    # Test wall warning (both channels, higher pitch)
    print("4. Playing WALL AHEAD warning (both channels, higher pitch)...")
    wall_beep = generate_beep(frequency=1000, channel='both')
    wall_beep.play()
    time.sleep(1.5)
    
    # Test sequence like in swimming
    print("5. Playing sequence: LEFT -> RIGHT -> WALL...")
    time.sleep(0.5)
    
    left_beep.play()
    time.sleep(2)
    
    right_beep.play()
    time.sleep(2)
    
    wall_beep.play()
    time.sleep(1.5)
    
    print("\nAudio test complete!")
    print("You should have heard:")
    print("- Startup beep in both ears")
    print("- GO LEFT beep in left ear only")
    print("- GO RIGHT beep in right ear only") 
    print("- WALL AHEAD beep in both ears (higher pitch)")
    print("- A sequence of LEFT -> RIGHT -> WALL")

if __name__ == "__main__":
    main() 