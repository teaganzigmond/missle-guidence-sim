import numpy as np

STRAIGHT_TIME1 = 25
CURVE_TIME = 25
STRAIGHT_TIME2 = 25

TARGET_SPEED = 750
MISSILE_SPEED = 1950

TURN_ANGLE = -np.pi * 4/3
YZ_ANGLE = -np.pi / 12

TMAX = 75
DT = 0.001

MISSILE_START = np.array([13000, 12000, 0], dtype=float)
AIRCRAFT_START = np.array([0, 0, 12000], dtype=float)

MISSILE_LAUNCH_TIME = 0 #seconds 
KILL_DISTANCE = 2

# Vertical drift rate during banked turn
# Adds realistic climb/descent through the turn maneuver
# Matches reference implementation in trace_and_chase.py
CLIMB_RATE_CURVE = -0.001

# Maximum allowable lateral acceleration in multiples of gravity
# Modern IR/radar-guided missiles typically operate 20-40 g
# Ref: MIL-HDBK-1211 (1995), Section 5
# MAX_G_FORCE = 30
MAX_G_FORCE = 50


# Gravitational acceleration constant
# Ref: U.S. Standard Atmosphere 1976 (NOAA/NASA), Table 2
GRAVITY = 9.80665  # m/s^2

MAX_ACCEL = MAX_G_FORCE * GRAVITY

# Proportional Navigation constant
# Typical effective navigation ratio: 3-5
# Ref: DTIC ADP010953, Section 3.2
NAV_CONSTANT = 5.0