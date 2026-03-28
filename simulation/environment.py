import numpy as np
from dataclasses import dataclass
from typing import Optional

import config
from simulation.target import Target
from simulation.missile import Missile, MissileState


@dataclass
class SimResult:
    """All outputs from a completed simulation run."""
    times           : np.ndarray        # (N,) time array
    target_positions: np.ndarray        # (N, 3) aircraft positions
    missile_positions: np.ndarray       # (N, 3) missile positions

    intercepted     : bool
    intercept_time  : Optional[float]   # None if missed
    intercept_index : Optional[int]     # Frame index of intercept
    intercept_pos   : Optional[np.ndarray]
    launch_time     : float

    @property
    def final_miss_distance(self) -> float:
        return float(np.linalg.norm(
            self.target_positions[-1] - self.missile_positions[-1]
        ))

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  SIMULATION SUMMARY",
            "=" * 52,
            f"  Missile launch time : {self.launch_time:.2f} s",
            f"  Target speed        : {config.TARGET_SPEED:.0f} m/s",
            f"  Missile speed       : {config.MISSILE_SPEED:.0f} m/s",
            f"  Nav constant (N)    : {config.NAV_CONSTANT}",
        ]
        if self.intercepted:
            lines += [
                f"  Result              : INTERCEPT",
                f"  Intercept time      : {self.intercept_time:.3f} s",
                f"  Intercept position  : "
                f"({self.intercept_pos[0]:.1f}, "
                f"{self.intercept_pos[1]:.1f}, "
                f"{self.intercept_pos[2]:.1f}) m",
            ]
        else:
            lines += [
                f"  Result              : MISS",
                f"  Final miss distance : {self.final_miss_distance:.1f} m",
            ]
        lines.append("=" * 52)
        return "\n".join(lines)


class Environment:
    """
    Orchestrates the simulation loop.

    Creates a Target and Missile, steps them forward in lockstep, and
    packages the results into a SimResult for analysis and animation.
    """

    def run(self) -> SimResult:
        """
        Execute the full simulation and return a SimResult.

        The target trajectory is precomputed by the Target class, so the
        loop only needs to step the missile forward each timestep.
        """
        target = Target()
        missile = Missile()

        times = target.times
        n     = len(times)

        # Target positions are already precomputed — no work needed here
        target_pos_arr = target.positions

        # Missile positions are built up one step at a time
        missile_pos_arr    = np.empty((n, 3))
        missile_pos_arr[0] = missile.position.copy()   # store initial position

        intercept_index = None  # frame index where kill condition was first met

        for i in range(1, n):
            t       = times[i]
            tgt_pos = target.position_at_index(i)
            tgt_vel = target.velocity_at_index(i)    # needed by PN guidance

            # Advance missile one timestep; returns current position
            mis_pos = missile.step(t, tgt_pos, tgt_vel, config.DT)
            missile_pos_arr[i] = mis_pos

            # Record the frame index the moment intercept is detected
            if missile.state == MissileState.INTERCEPTED and intercept_index is None:
                intercept_index = i

        # If the loop ended without an intercept, formally mark as a miss
        missile.mark_missed()

        intercepted = (missile.state == MissileState.INTERCEPTED)

        return SimResult(
            times             = times,
            target_positions  = target_pos_arr,
            missile_positions = missile_pos_arr,
            intercepted       = intercepted,
            intercept_time    = missile.intercept_time,
            intercept_index   = intercept_index,
            intercept_pos     = missile.intercept_pos,
            # Fall back to config value if missile never actually launched
            launch_time       = missile.launch_time
                                if missile.launch_time is not None
                                else config.MISSILE_LAUNCH_TIME,
        )