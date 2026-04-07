import numpy as np
import config
from guidance.proportional_navigation import proportional_navigation

# MissileState enum for clean state machine
# Ref: MIL-HDBK-1211(MI), Section 5.5 - missile flight phases
from enum import Enum

class MissileState(Enum):
    FLYING   = "flying"
    HIT      = "hit"
    MISSED   = "missed"


class Missile:

    def __init__(self):
        self.position = config.MISSILE_START.copy().astype(float)
        self.speed    = config.MISSILE_SPEED

        # Initialize velocity as a unit vector pointing generally toward
        # the aircraft start, scaled to missile speed.
        # This gives PN a meaningful initial velocity vector to work with.
        # Ref: MIL-HDBK-1211(MI) Section 4.4.1 - initial conditions
        initial_direction = config.AIRCRAFT_START - config.MISSILE_START
        norm = np.linalg.norm(initial_direction)
        if norm > 0:
            self.velocity = (initial_direction / norm) * self.speed
        else:
            self.velocity = np.array([self.speed, 0.0, 0.0])

        self.state   = MissileState.FLYING
        self.peak_g  = 0.0

    @property
    def active(self):
        return self.state == MissileState.FLYING

    def step(self, target_pos, target_vel, dt):
        """
        Advance the missile state by one timestep.

        PARAMETERS
        ----------
        target_pos : np.array (3,)
            Current target position [m]
        target_vel : np.array (3,)
            Current target velocity [m/s]
        dt : float
            Simulation timestep [s]

        RETURNS
        -------
        position : np.array (3,)
            Updated missile position
        """

        # Missile is frozen once it has hit or missed
        if not self.active:
            return self.position

        # --------------------------------------------------
        # Check intercept condition
        # Ref: MIL-HDBK-1211(MI) Section 6.2.3
        # --------------------------------------------------
        r = target_pos - self.position
        distance = np.linalg.norm(r)

        if distance < config.KILL_DISTANCE:
            print(f"  [HIT] distance = {distance:.2f} m")
            self.state = MissileState.HIT
            return self.position

        # --------------------------------------------------
        # Proportional Navigation guidance
        # Ref: DTIC ADP010953
        # --------------------------------------------------
        accel = proportional_navigation(
        self.position,
        self.velocity,
        target_pos,
        target_vel,
        N=config.NAV_CONSTANT
        )

        # --------------------------------------------------
        # Lateral g-force limiting
        # Only lateral (perpendicular to velocity) acceleration
        # is limited — axial thrust is separate.
        # Ref: MIL-HDBK-1211(MI) Section 5.6.2
        # --------------------------------------------------
        vel_hat = self.velocity / np.linalg.norm(self.velocity)
        accel_lateral = accel - np.dot(accel, vel_hat) * vel_hat
        lateral_mag   = np.linalg.norm(accel_lateral)

        if lateral_mag > config.MAX_ACCEL:
            accel_lateral = accel_lateral * (config.MAX_ACCEL / lateral_mag)

        # Track peak g-load (lateral only)
        # Ref: MIL-HDBK-1211(MI) Section 5.6
        current_g = lateral_mag / config.GRAVITY
        if current_g > self.peak_g:
            self.peak_g = current_g

        # --------------------------------------------------
        # Update velocity: apply lateral acceleration only
        # Speed is held constant by propulsion (constant-speed
        # assumption for boost-sustain motors)
        # Ref: MIL-HDBK-1211(MI) Section 5.4
        # --------------------------------------------------
        self.velocity = self.velocity + accel_lateral * dt

        # Re-normalize to maintain constant speed
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            self.velocity = (self.velocity / speed) * self.speed

        # --------------------------------------------------
        # Update position: x = x + v*dt
        # --------------------------------------------------
        self.position = self.position + self.velocity * dt

        return self.position