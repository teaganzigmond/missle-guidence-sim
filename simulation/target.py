import numpy as np
import config


class Target:
    """
    Models a maneuvering aircraft flying a three-segment trajectory:
      1. Straight flight along +X for STRAIGHT_TIME1 seconds
      2. A banked circular turn with vertical drift over CURVE_TIME seconds
      3. Straight flight in the post-turn direction for STRAIGHT_TIME2 seconds,
         after which the target holds its final position

    Trajectory geometry matches the reference implementation in
    trace_and_chase.py (read-only).
    """

    def __init__(self):
        self.start = config.AIRCRAFT_START.copy()
        self.vel = config.TARGET_SPEED

        # Lazily initialized on first entry into each segment
        self._curve_start = None
        self._straight2_start = None
        self._center = None
        self._radius = None

        # Cached final position so we can hold it after STRAIGHT_TIME2
        self._final_pos = None

    def position(self, t):
        v = self.vel

        # ----------------------------------------------------------
        # Segment 1: straight flight in +X direction
        # ----------------------------------------------------------
        if t <= config.STRAIGHT_TIME1:
            return np.array([
                self.start[0] + v * t,
                self.start[1],
                self.start[2]
            ])

        # ----------------------------------------------------------
        # Segment 2: banked circular turn with vertical drift
        #
        # CLIMB_RATE_CURVE adds a realistic vertical displacement
        # during the turn, matching the reference implementation.
        # The drift uses a cosine envelope so it starts and ends
        # smoothly at zero rate (no discontinuity at segment joins).
        #
        # Ref: trace_and_chase.py target_location(), curve segment
        # ----------------------------------------------------------
        elif t <= config.STRAIGHT_TIME1 + config.CURVE_TIME:
            tc = t - config.STRAIGHT_TIME1

            if self._curve_start is None:
                self._curve_start = np.array([
                    self.start[0] + v * config.STRAIGHT_TIME1,
                    self.start[1],
                    self.start[2]
                ])
                self._radius = (v * config.CURVE_TIME) / config.TURN_ANGLE
                self._center = np.array([
                    self._curve_start[0],
                    self._curve_start[1] + self._radius * np.cos(config.YZ_ANGLE),
                    self._curve_start[2] + self._radius * np.sin(config.YZ_ANGLE)
                ])

            angle = tc * config.TURN_ANGLE / config.CURVE_TIME
            arc_angle = -np.pi / 2 + angle

            # Vertical drift envelope: smooth cosine ramp
            # Matches trace_and_chase.py exactly
            drift = v**2 * (1 - np.cos(np.pi * tc / config.CURVE_TIME)) * config.CLIMB_RATE_CURVE

            return np.array([
                self._center[0] + self._radius * np.cos(arc_angle),
                self._center[1] + self._radius * np.sin(arc_angle) * np.cos(config.YZ_ANGLE)
                             + np.cos(config.YZ_ANGLE + np.pi / 2) * drift,
                self._center[2] + self._radius * np.sin(arc_angle) * np.sin(config.YZ_ANGLE)
                             + np.sin(config.YZ_ANGLE + np.pi / 2) * drift
            ])

        # ----------------------------------------------------------
        # Segment 3: straight flight in post-turn direction
        # Target holds final position after STRAIGHT_TIME2 elapses
        # ----------------------------------------------------------
        else:
            if self._straight2_start is None:
                # Evaluate exact curve-end position for a clean join
                self._straight2_start = self.position(
                    config.STRAIGHT_TIME1 + config.CURVE_TIME
                )

            ts = min(t - (config.STRAIGHT_TIME1 + config.CURVE_TIME),
                     config.STRAIGHT_TIME2)

            dx = np.cos(config.TURN_ANGLE)
            dy = np.sin(config.TURN_ANGLE) * np.cos(config.YZ_ANGLE)
            dz = np.sin(config.TURN_ANGLE) * np.sin(config.YZ_ANGLE)

            return np.array([
                self._straight2_start[0] + v * ts * dx,
                self._straight2_start[1] + v * ts * dy,
                self._straight2_start[2] + v * ts * dz
            ])