import numpy as np

STRAIGHT_TIME1 = 25
CURVE_TIME = 25
STRAIGHT_TIME2 = 25

# ---------------------------------------------------------------------------
# Target and missile speeds
#
# TARGET_SPEED: 750 m/s ≈ Mach 2.2 at sea level — representative of a
# fast supersonic fighter (e.g. MiG-25 Foxbat top speed ~Mach 2.8,
# F-15 Eagle top speed ~Mach 2.5). Realistic for a high-performance
# evading aircraft.
#
# MISSILE_SPEED: 1374 m/s ≈ Mach 4.0 at sea level — matches published
# top speed of AIM-120 AMRAAM (Mach 4 capable).
# Ref: Jane's Air-Launched Weapons; globalsecurity.org AIM-120 specs
# Speed ratio ~1.83:1 (missile:target) — realistic for BVR engagement.
# ---------------------------------------------------------------------------
TARGET_SPEED = 857.5    # m/s — Mach ~2.5 at altitude, MiG-25 operational cruise speed
# TARGET_SPEED = 754.6    # m/s — Mach ~2.2 at sea level
MISSILE_SPEED = 1374  # m/s — Mach ~4.0 at sea level (AIM-120 AMRAAM class)

# ---------------------------------------------------------------------------
# Missile physical properties
# Based on AIM-120 AMRAAM class tactical missile
# Ref: Jane's Air-Launched Weapons; globalsecurity.org AIM-120 specs
# ---------------------------------------------------------------------------
MISSILE_MASS = 152.0   # kg — launch mass (fixed, fuel burn not modeled at this stage)

# ---------------------------------------------------------------------------
# Motor boost phase
# SAM launches from dead stop — boost phase accelerates from 0 to MISSILE_SPEED
# Boost duration ~3s is representative of tactical SAM solid-fuel boosters
# During boost, PN guidance is inactive — missile flies a preset vertical climb
# Ref: Jane's Land-Based Air Defence; MIL-HDBK-1211(MI) Section 5.4
# ---------------------------------------------------------------------------
BOOST_TIME = 3.0                            # seconds
BOOST_ACCEL = MISSILE_SPEED / BOOST_TIME    # m/s² — linear ramp assumption
                                             # Ref: MIL-HDBK-1211(MI) Section 5.4
# ---------------------------------------------------------------------------
# Target evasion parameters
# Models a fighter pilot responding to a missile warning
#
# Detection range: modern RWR (Radar Warning Receiver) systems can detect
# missile launch at ~25-30km. We use 25km as conservative terminal detection.
# Ref: Jane's Avionics — Radar Warning Receivers
#
# Reaction time: ~0.5s accounts for pilot recognition and control input
# Ref: MIL-HDBK-1472 — Human factors, pilot reaction time
#
# Max g: F-15/F-16 class fighter sustained turn ~9g
# Max speed: full afterburner ~900 m/s (Mach ~2.6)
# ---------------------------------------------------------------------------
TARGET_DETECTION_RANGE = 25000.0   # m — missile detection range
TARGET_REACTION_TIME   = 0.5      # s — pilot reaction delay
TARGET_MAX_G           = 9.0      # g — max sustained evasive turn
TARGET_MAX_SPEED       = 1000    # m/s — max speed in afterburner
# TARGET_MAX_SPEED       = 900.0    # m/s — max speed in afterburner


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
MAX_G_FORCE = 40


# Gravitational acceleration constant
# Ref: U.S. Standard Atmosphere 1976 (NOAA/NASA), Table 2
GRAVITY = 9.80665  # m/s^2

MAX_ACCEL = MAX_G_FORCE * GRAVITY

# Proportional Navigation constant
# Typical effective navigation ratio: 3-5
# Ref: DTIC ADP010953, Section 3.2
NAV_CONSTANT = 5.0