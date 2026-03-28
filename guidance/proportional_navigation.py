import numpy as np


def proportional_navigation(
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    N: float = 3.0,
) -> np.ndarray:
    """
    Proportional Navigation Guidance Law.

    Commands a lateral acceleration proportional to the rotation rate of
    the Line-Of-Sight (LOS) vector, scaled by closing velocity and the
    navigation constant N.

        a_cmd = N · Vc · ω_LOS

    where ω_LOS is the LOS angular rate vector and Vc is closing speed.

    Parameters
    ----------
    missile_pos : (3,) array — missile position (m)
    missile_vel : (3,) array — missile velocity vector (m/s)
    target_pos  : (3,) array — target position (m)
    target_vel  : (3,) array — target velocity vector (m/s)
    N           : navigation constant (dimensionless, typically 3–5)

    Returns
    -------
    accel_cmd : (3,) array — commanded acceleration (m/s²)
    """
    # Relative position (LOS vector: missile → target)
    r = target_pos - missile_pos
    r_mag = np.linalg.norm(r)

    if r_mag < 1e-6:
        return np.zeros(3)

    # Relative velocity
    v_rel = target_vel - missile_vel

    # LOS angular rate vector  ω = (r × v) / |r|²
    los_rate = np.cross(r, v_rel) / (r_mag ** 2)

    # Closing speed  Vc = -(r̂ · v_rel)
    closing_vel = -np.dot(r, v_rel) / r_mag

    # PN command
    accel_cmd = N * closing_vel * los_rate

    return accel_cmd