import numpy as np

STRAIGHT_TIME1 = 25
CURVE_TIME = 25
STRAIGHT_TIME2 = 25

TARGET_SPEED = 750
MISSILE_SPEED = 800

TURN_ANGLE = -np.pi * 4/3
YZ_ANGLE = -np.pi / 12

TMAX = 75
DT = 0.001

MISSILE_START = np.array([13000,12000,0])
AIRCRAFT_START = np.array([0,0,12000])

MISSILE_LAUNCH_TIME = 0
KILL_DISTANCE = 2
CLIMB_RATE_CURVE = -0.001

#######################################
#   Missle G-Force and Gravity
#######################################
# Maximum allowable acceleration in multiples of gravity
# (modern missiles often operate around 20–40 g)
MAX_G_FORCE = 30

# Gravitational acceleration constant
GRAVITY = 9.81  # m/s^2

# Convert g-force limit to acceleration limit
MAX_ACCEL = MAX_G_FORCE * GRAVITY