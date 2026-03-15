import numpy as np
import config
from guidance import proportional_navigation

class Missile:

    def __init__(self):

        self.position = config.MISSILE_START.copy()
        self.velocity = config.MISSILE_SPEED
        self.alive = True

    def step(self, target_pos, target_vel, dt):
        """
        Advance the missile state by one timestep.

        PARAMETERS
        ----------
        target_pos : np.array
            Current target position

        target_vel : np.array
            Current target velocity

        dt : float
            Simulation timestep
        """

        # If the missile has already intercepted the target
        # we stop updating its position.
        if not self.active:
            return self.position

        # --------------------------------------------------
        # Compute distance to target
        # --------------------------------------------------
        r = target_pos - self.position
        distance = np.linalg.norm(r)

        # --------------------------------------------------
        # Check for intercept condition
        #
        # If the missile is within the kill radius we
        # consider the target destroyed.
        # --------------------------------------------------
        if distance < config.KILL_DISTANCE:
            print("Intercept achieved!")
            self.active = False
            return self.position

        # --------------------------------------------------
        # Compute guidance acceleration using
        # Proportional Navigation
        # --------------------------------------------------
        accel = proportional_navigation(
            self.position,
            self.velocity,
            target_pos,
            target_vel
        )

    # --------------------------------------------------
    # Update missile velocity
    #
    # v = v + a * dt
    # --------------------------------------------------
        self.velocity += accel * dt

    # --------------------------------------------------
    # Maintain constant missile speed
    #
    # Real missiles try to maintain near-constant
    # speed using propulsion.
    # --------------------------------------------------
        speed = np.linalg.norm(self.velocity)

        if speed > 0:
            self.velocity = (self.velocity / speed) * config.MISSILE_SPEED

    # --------------------------------------------------
    # Update position using velocity
    #
    # x = x + v * dt
    # --------------------------------------------------
        self.position += self.velocity * dt

        return self.position
    """
    WANT TO ADD
    proportional_navigation()
    ai_guidance()
    """