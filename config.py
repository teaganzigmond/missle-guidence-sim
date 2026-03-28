import numpy as np

# ============================================================================
# Time parameters
# ============================================================================
STRAIGHT_TIME1   = 25       # Duration of first straight segment (s)
CURVE_TIME       = 25       # Duration of turn (s)
STRAIGHT_TIME2   = 25       # Duration of second straight segment (s)
TMAX             = 75       # Total simulation time (s)
DT               = 0.001    # Timestep (s)

# ============================================================================
# Speeds
# ============================================================================
TARGET_SPEED     = 750      # Aircraft speed (m/s)
MISSILE_SPEED    = 800      # Missile speed (m/s)

# ============================================================================
# Geometry
# ============================================================================
TURN_ANGLE       = -np.pi * 4 / 3   # Aircraft turn angle (rad)
YZ_ANGLE         = -np.pi / 12      # Out-of-plane tilt of turn (rad)
CLIMB_RATE_CURVE = -0.001            # Altitude change rate during curve

# ============================================================================
# Starting positions
# ============================================================================
MISSILE_START    = np.array([13000.0, 12000.0,     0.0])   # (m)
AIRCRAFT_START   = np.array([    0.0,     0.0, 12000.0])   # (m)

# ============================================================================
# Missile parameters
# ============================================================================
MISSILE_LAUNCH_TIME = 0.0   # Time missile launches (s)
KILL_DISTANCE       = 2.0   # Intercept radius (m)

# ============================================================================
# Guidance
# ============================================================================
NAV_CONSTANT = 3.0          # Proportional Navigation constant (typically 3–5)

# ============================================================================
# Animation
# ============================================================================
ANIMATION_INTERVAL  = 5     # Frame interval (ms)
MAX_ANIMATION_FRAMES = 500  # Cap on number of rendered frames