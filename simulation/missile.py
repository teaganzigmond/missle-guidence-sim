import numpy as np
from enum import Enum, auto

import config
from guidance.proportional_navigation import proportional_navigation


class MissileState(Enum):
    """Lifecycle states of the missile."""
    READY       = auto()   # Waiting for launch time
    LAUNCHED    = auto()   # In flight, homing on target
    INTERCEPTED = auto()   # Kill condition met
    MISSED      = auto()   # Simulation ended without intercept


class Missile:
    """
    Proportional Navigation missile that homes on a moving target.

    The missile maintains a velocity *vector* (not just a scalar speed)
    so that PN guidance can apply lateral acceleration correctly.
    On each timestep the speed is renormalised to MISSILE_SPEED so that
    propulsion is assumed to hold Mach number constant.
    """

    def __init__(self):
        self.state          = MissileState.READY
        self.position       = config.MISSILE_START.astype(float)

        # Initialise velocity pointing directly at aircraft start
        initial_dir         = config.AIRCRAFT_START - config.MISSILE_START
        norm                = np.linalg.norm(initial_dir)
        if norm > 0:
            initial_dir = initial_dir / norm
        self.velocity       = initial_dir * config.MISSILE_SPEED

        # Event records
        self.intercept_time  = None
        self.intercept_pos   = None
        self.launch_time     = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.state == MissileState.LAUNCHED

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self,
             t: float,
             target_pos: np.ndarray,
             target_vel: np.ndarray,
             dt: float) -> np.ndarray:
        """
        Advance missile state by one timestep.

        Parameters
        ----------
        t          : current simulation time (s)
        target_pos : target position this timestep (m)
        target_vel : target velocity this timestep (m/s)
        dt         : timestep (s)

        Returns
        -------
        Current missile position (m).
        """

        # ── Waiting to launch ─────────────────────────────────────────
        if self.state == MissileState.READY:
            if t >= config.MISSILE_LAUNCH_TIME:
                self.state      = MissileState.LAUNCHED
                self.launch_time = t
            return self.position.copy()

        # ── Already terminal ──────────────────────────────────────────
        if self.state in (MissileState.INTERCEPTED, MissileState.MISSED):
            return self.position.copy()

        # ── In flight ─────────────────────────────────────────────────

        # Check intercept condition before moving
        r        = target_pos - self.position
        distance = np.linalg.norm(r)

        if distance < config.KILL_DISTANCE:
            self.state         = MissileState.INTERCEPTED
            self.intercept_time = t
            self.intercept_pos  = self.position.copy()
            return self.position.copy()

        # Compute PN acceleration
        accel = proportional_navigation(
            missile_pos=self.position,
            missile_vel=self.velocity,
            target_pos=target_pos,
            target_vel=target_vel,
            N=config.NAV_CONSTANT,
        )

        # Update velocity: v = v + a·dt
        self.velocity += accel * dt

        # Renormalise to constant speed (propulsion holds Mach)
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            self.velocity = (self.velocity / speed) * config.MISSILE_SPEED

        # Update position: x = x + v·dt
        self.position += self.velocity * dt

        return self.position.copy()

    def mark_missed(self):
        """Call after simulation loop ends if no intercept occurred."""
        if self.state == MissileState.LAUNCHED:
            self.state = MissileState.MISSED