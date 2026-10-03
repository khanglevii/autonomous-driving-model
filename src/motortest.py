import gpiozero as gpio
import numpy as np
from time import sleep

class motorctrl:
    def __init__(engine, In1A, In2A, In1B, In2B):
        # Set up GPIO pins
        engine.left = gpio.Motor(forward=In2A,backward=In2B,pwm=True)
        engine.right = gpio.Motor(forward=In1A,backward=In1B,pwm=True)
        
        #Start both motors at 0%
        engine.left.forward(0)
        engine.right.forward(0)

    def move(engine, speed=0, maxspeed=0, turn=0):  # Speed is duty cycle (0-100%)
        leftSpeed = speed - turn
        rightSpeed = speed + turn
        
        leftSpeed = np.clip(leftSpeed,0,maxspeed)
        rightSpeed = np.clip(rightSpeed,0,maxspeed)
        leftSpeed = np.interp(leftSpeed,[0,255],[0,1])
        rightSpeed = np.interp(rightSpeed,[0,255],[0,1])
        
        # Change the duty cycle for both motors
        engine.left.forward(leftSpeed)
        engine.right.forward(rightSpeed)
        return 0

    def rev(engine):
        engine.left.reverse(0.3)
        engine.right.reverse(0.3)
        return 0
        
    def stop(engine):
        # Stop PWM and clean up GPIO
        engine.left.stop()
        engine.right.stop()
        return 0
