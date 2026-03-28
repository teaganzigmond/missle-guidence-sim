import numpy as np
import config


class Target:
    """
    Aircraft flying a three-segment trajectory:
      1. Straight flight in +X from AIRCRAFT_START
      2. Circular arc turn (optionally tilted out-of-plane)
      3. Straight flight in the post-turn heading

    The full trajectory is precomputed once on construction so that
    position can be queried at any time index without mutable global state.
    """

    def __init__(self):
        self._times         = np.arange(0, config.TMAX, config.DT)
        self._positions     = self._precompute()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def positions(self) -> np.ndarray:
        """Shape (N, 3) array of [x, y, z] positions."""
        return self._positions

    def position_at_index(self, i: int) -> np.ndarray:
        """Return the precomputed (3,) position vector at timestep i."""
        return self._positions[i]

    def velocity_at_index(self, i: int) -> np.ndarray:
        """
        Finite-difference velocity estimate (m/s).
        Uses forward difference except at the last point.
        """
        if i < len(self._times) - 1:
            dp = self._positions[i + 1] - self._positions[i]
            return dp / config.DT
        else:
            dp = self._positions[i] - self._positions[i - 1]
            return dp / config.DT

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute(self) -> np.ndarray:
        """
        Build the full (N, 3) position array by walking through
        each time segment analytically.
        """
        vel        = config.TARGET_SPEED
        t1         = config.STRAIGHT_TIME1
        tc         = config.CURVE_TIME
        t2         = config.STRAIGHT_TIME2
        theta      = config.TURN_ANGLE
        phi        = config.YZ_ANGLE          # yz tilt angle
        climb      = config.CLIMB_RATE_CURVE
        start      = config.AIRCRAFT_START.astype(float)

        # Arc radius (negative turn_angle → negative radius handled by sign)
        radius = (vel * tc) / theta

        # --- Segment boundary positions ---
        # End of straight 1 / start of curve
        curve_start = start + np.array([vel * t1, 0.0, 0.0])

        # Arc centre (perpendicular to initial heading, tilted by phi)
        arc_centre = curve_start + np.array([
            0.0,
            radius * np.cos(phi),
            radius * np.sin(phi),
        ])

        # End of curve / start of straight 2
        # At tc the arc_angle = -pi/2 + theta
        final_arc_angle = -np.pi / 2 + theta
        climb_offset    = vel**2 * (1 - np.cos(np.pi)) * climb   # (1-cos(π)) = 2

        curve_end = np.array([
            arc_centre[0] + radius * np.cos(final_arc_angle),
            arc_centre[1] + radius * np.sin(final_arc_angle) * np.cos(phi)
                           + np.cos(phi + np.pi / 2) * climb_offset,
            arc_centre[2] + radius * np.sin(final_arc_angle) * np.sin(phi)
                           + np.sin(phi + np.pi / 2) * climb_offset,
        ])

        # Post-turn unit direction vector
        post_turn_dir = np.array([
            np.cos(theta),
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
        ])

        # --- Fill position array ---
        times  = self._times
        n      = len(times)
        pos    = np.empty((n, 3))

        for i, t in enumerate(times):
            if t <= t1:
                # Segment 1: straight in +X
                pos[i] = start + np.array([vel * t, 0.0, 0.0])

            elif t <= t1 + tc:
                # Segment 2: circular arc with optional climb
                tc_local   = t - t1
                arc_angle  = -np.pi / 2 + tc_local * theta / tc
                climb_disp = vel**2 * (1 - np.cos(np.pi * tc_local / tc)) * climb

                pos[i] = np.array([
                    arc_centre[0] + radius * np.cos(arc_angle),
                    arc_centre[1] + radius * np.sin(arc_angle) * np.cos(phi)
                                  + np.cos(phi + np.pi / 2) * climb_disp,
                    arc_centre[2] + radius * np.sin(arc_angle) * np.sin(phi)
                                  + np.sin(phi + np.pi / 2) * climb_disp,
                ])

            elif t <= t1 + tc + t2:
                # Segment 3: straight in post-turn direction
                ts     = t - (t1 + tc)
                pos[i] = curve_end + vel * ts * post_turn_dir

            else:
                # Beyond TMAX — hold last position
                pos[i] = pos[i - 1]

        return pos