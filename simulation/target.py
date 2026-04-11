from turtle import pos

import numpy as np
from torch import dist
import config


class Target:
    """
    Models a maneuvering aircraft flying a three-segment trajectory,
    with missile-aware evasive maneuver capability.

    Normal flight: three-segment preset path (straight → banked turn → straight)

    Evasion — hard break arc:
    Once the missile is detected within TARGET_DETECTION_RANGE, after a
    reaction delay the target executes a hard 120° banked turn away from
    the missile. The arc is constructed in velocity-space so it works
    correctly regardless of which direction the target is flying.
    After the arc, the target flies straight at afterburner speed.

    Ref: Shaw, R.L. — Fighter Combat: Tactics and Maneuvering (1985)
    Ref: Zarchan — Tactical and Strategic Missile Guidance (2012)
    Ref: trace_and_chase.py — reference arc geometry (read-only)
    """

    def __init__(self):
        self.start = config.AIRCRAFT_START.copy()
        self.vel   = config.TARGET_SPEED

        # Segment initialization flags
        self._curve_start     = None
        self._straight2_start = None
        self._center          = None
        self._radius          = None

        # Evasion state
        self._evading           = False
        self._detection_time    = None
        self._evasion_start_t   = None
        self._evasion_forward   = None   # unit velocity at evasion start
        self._evasion_break     = None   # unit break direction
        self._evasion_center    = None   # center of arc
        self._evasion_radius    = None   # arc radius
        self._evasion_angle     = None   # total turn angle
        self._post_arc_pos      = None   # pre-computed arc end position
        self._post_arc_dir      = None   # pre-computed arc end direction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def position(self, t, missile_pos=None):
        """
        Calculate target position at time t.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        missile_pos : np.ndarray (3,) or None
            Current missile position [m].
        """

        if not self._evading:
            pos = self._preset_position(t)

            if missile_pos is not None and t > config.MISSILE_LAUNCH_TIME + config.BOOST_TIME:
                dist = np.linalg.norm(missile_pos - pos)
                if dist < config.TARGET_DETECTION_RANGE:
                    if self._detection_time is None:
                        self._detection_time = t
                        print(f"  [TARGET] Missile detected at"
                              f" {dist/1000:.1f} km, t={t:.2f}s")

                    if t - self._detection_time >= config.TARGET_REACTION_TIME:
                        self._init_evasion(t, pos, missile_pos)

            return pos

        return self._evasion_position(t)

    # ------------------------------------------------------------------
    # Evasion initialization
    # ------------------------------------------------------------------

    def _init_evasion(self, t, pos, missile_pos):
        """
        Set up evasion arc in velocity-space coordinates.

        Uses nominal TARGET_SPEED (not finite-diff magnitude) to avoid
        near-zero speed at segment boundaries. Arc center is pre-computed
        so the straight phase never sees None.
        """
        self._evading        = True
        self._evasion_start_t = t

        # ----------------------------------------------------------
        # Current velocity direction from finite difference
        # Use nominal speed to avoid magnitude errors at boundaries
        # ----------------------------------------------------------
        t_prev = max(t - config.DT, 0.0)
        prev   = self._preset_position(t_prev)
        vel    = pos - prev
        speed  = np.linalg.norm(vel)

        if speed < 1.0:
            # Fallback — target is barely moving, use +X
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = vel / speed

        self._evasion_forward = forward

        # ----------------------------------------------------------
        # Break direction — perpendicular to forward, away from missile
        # ----------------------------------------------------------
        world_up = np.array([0.0, 0.0, 1.0])
        right    = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right = right / np.linalg.norm(right)

        # Missile direction from target
        to_missile = missile_pos - pos
        to_missile_norm = np.linalg.norm(to_missile)
        if to_missile_norm > 1e-6:
            to_missile = to_missile / to_missile_norm
        else:
            to_missile = -forward

        # Break away from missile
        if np.dot(right, to_missile) > 0:
            break_dir = -right   # missile right → break left
        else:
            break_dir = right    # missile left  → break right

        self._evasion_break = break_dir

        # ----------------------------------------------------------
        # Arc geometry using nominal TARGET_SPEED
        #
        # r = v² / (n*g)
        # Ref: MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft
        # ----------------------------------------------------------
        nominal_speed         = config.TARGET_SPEED
        self._evasion_radius  = (nominal_speed**2
                                 / (config.TARGET_MAX_G * config.GRAVITY))
        self._evasion_angle   = np.pi * 2.0 / 3.0   # 120° break turn
        self._evasion_center  = pos + break_dir * self._evasion_radius

        # ----------------------------------------------------------
        # Pre-compute post-arc state so straight phase is always valid
        # ----------------------------------------------------------
        end_theta = self._evasion_angle
        r         = self._evasion_radius
        ctr       = self._evasion_center

        self._post_arc_pos = (ctr
                              - break_dir * r * np.cos(end_theta)
                              + forward   * r * np.sin(end_theta))

        end_dir = (forward   * np.cos(end_theta)
                   + break_dir * np.sin(end_theta))
        norm = np.linalg.norm(end_dir)
        self._post_arc_dir = end_dir / norm if norm > 1e-6 else forward

        side = 'left' if np.dot(break_dir, right) < 0 else 'right'
        print(f"  [TARGET] Evasion arc at t={t:.2f}s"
              f" | r={self._evasion_radius/1000:.1f}km"
              f" | {side} break")

    # ------------------------------------------------------------------
    # Evasion position
    # ------------------------------------------------------------------

    def _evasion_position(self, t):
        """
        Position along the break arc, then straight at afterburner.

        Arc phase:
            pos(θ) = center - break*r*cos(θ) + forward*r*sin(θ)
            At θ=0: pos = center - break*r = start_pos  ✓
            Angular rate: ω = v/r = TARGET_MAX_G*g / v

        Straight phase:
            Afterburner speed ramp: TARGET_SPEED → TARGET_MAX_SPEED over 3s
            Position integrated analytically.
        """
        te      = t - self._evasion_start_t
        speed   = config.TARGET_SPEED
        radius  = self._evasion_radius
        omega   = speed / radius
        arc_time = self._evasion_angle / omega

        forward   = self._evasion_forward
        break_dir = self._evasion_break
        center    = self._evasion_center

        if te <= arc_time:
            # Arc phase
            theta = omega * te
            pos   = (center
                     - break_dir * radius * np.cos(theta)
                     + forward   * radius * np.sin(theta))
            return pos

        else:
            # Straight phase — afterburner speed ramp
            ts        = te - arc_time
            ramp_time = 3.0
            v0        = config.TARGET_SPEED
            v1        = config.TARGET_MAX_SPEED

            if ts <= ramp_time:
                dist = v0 * ts + (v1 - v0) * ts**2 / (2.0 * ramp_time)
            else:
                dist_ramp     = v0 * ramp_time + (v1 - v0) * ramp_time / 2.0
                dist_straight = v1 * (ts - ramp_time)
                dist          = dist_ramp + dist_straight

            return self._post_arc_pos + self._post_arc_dir * dist

    # ------------------------------------------------------------------
    # Preset trajectory
    # ------------------------------------------------------------------

    def _preset_position(self, t):
        """
        Original three-segment preset trajectory.
        Matches trace_and_chase.py reference exactly.
        """
        v = config.TARGET_SPEED

        if t <= config.STRAIGHT_TIME1:
            return np.array([
                self.start[0] + v * t,
                self.start[1],
                self.start[2]
            ])

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

            angle     = tc * config.TURN_ANGLE / config.CURVE_TIME
            arc_angle = -np.pi / 2 + angle
            drift     = (v**2
                         * (1 - np.cos(np.pi * tc / config.CURVE_TIME))
                         * config.CLIMB_RATE_CURVE)

            return np.array([
                self._center[0] + self._radius * np.cos(arc_angle),
                self._center[1] + self._radius * np.sin(arc_angle) * np.cos(config.YZ_ANGLE)
                             + np.cos(config.YZ_ANGLE + np.pi / 2) * drift,
                self._center[2] + self._radius * np.sin(arc_angle) * np.sin(config.YZ_ANGLE)
                             + np.sin(config.YZ_ANGLE + np.pi / 2) * drift
            ])

        else:
            if self._straight2_start is None:
                self._straight2_start = self._preset_position(
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