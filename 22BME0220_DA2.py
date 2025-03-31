# %%
try:
    import RPi.GPIO as GPIO  # Real library (for Raspberry Pi)
except ModuleNotFoundError:
    from FakeRPi import GPIO  # Fake library (for Windows/Mac) 
import time  
import random  # Simulating sensor values (replace with real sensor readings)

# Define GPIO pins
SPRINKLER_PIN = 23  
GPIO.setmode(GPIO.BCM)  
GPIO.setup(SPRINKLER_PIN, GPIO.OUT)  

# Function to read simulated sensor values
def get_temperature():
    return random.randint(30, 80)  # Simulated temperature reading

def get_smoke_level():
    return random.randint(100, 500)  # Simulated smoke sensor reading

# AI-based rule system to detect fire
def check_fire():
    temp = get_temperature()
    smoke = get_smoke_level()
    print(f"Temperature: {temp}°C, Smoke Level: {smoke} PPM")
    
    if temp > 60 or smoke > 300:  # Fire condition
        print("🔥 Fire detected! Activating sprinkler...")
        GPIO.output(SPRINKLER_PIN, GPIO.HIGH)  # Turn on sprinkler
        time.sleep(5)  # Sprinkler on for 5 seconds
        GPIO.output(SPRINKLER_PIN, GPIO.LOW)   # Turn off sprinkler
        print("✅ Fire extinguished, sprinkler off.")

# Run system
while True:
    check_fire()
    time.sleep(2)  # Check every 2 seconds


