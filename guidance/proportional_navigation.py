import numpy as np


def proportional_navigation(missile_pos,
                            missile_vel,
                            target_pos,
                            target_vel,
                            N=3):
    """
    Proportional Navigation Guidance

    This function computes the lateral acceleration that the missile should
    apply in order to intercept the target.

    PARAMETERS
    ----------
    missile_pos : np.array (3D vector)
        Current position of the missile [x, y, z]

    missile_vel : np.array (3D vector)
        Current velocity vector of the missile

    target_pos : np.array (3D vector)
        Current position of the target

    target_vel : np.array (3D vector)
        Current velocity vector of the target

    N : float
        Navigation constant (typically between 3 and 5)

    RETURNS
    -------
    accel_command : np.array
        Acceleration vector the missile should apply
    """

    # ----------------------------------------------------------
    # Step 1: Compute relative position vector
    #
    # This is the vector from the missile to the target.
    # It defines the "Line Of Sight" (LOS).
    # ----------------------------------------------------------
    r = target_pos - missile_pos

    # ----------------------------------------------------------
    # Step 2: Compute relative velocity
    #
    # This tells us how the target is moving relative
    # to the missile.
    # ----------------------------------------------------------
    v = target_vel - missile_vel

    # ----------------------------------------------------------
    # Step 3: Compute Line-of-Sight (LOS) rotation rate
    #
    # If the LOS vector rotates over time, the target
    # is not on a collision course.
    #
    # The cross product captures the angular rate
    # of this LOS rotation.
    # ----------------------------------------------------------
    los_rate = np.cross(r, v) / np.linalg.norm(r)**2

    # ----------------------------------------------------------
    # Step 4: Compute closing velocity
    #
    # This measures how fast the missile and target
    # are moving toward each other.
    #
    # Positive closing velocity means interception
    # is possible.
    # ----------------------------------------------------------
    closing_vel = -np.dot(r, v) / np.linalg.norm(r)

    # ----------------------------------------------------------
    # Step 5: Compute commanded acceleration
    #
    # Proportional Navigation Law:
    #
    #   a = N * Vc * (LOS rate)
    #
    # where:
    #   N  = navigation constant
    #   Vc = closing velocity
    #
    # This causes the missile to steer in a way that
    # keeps the LOS angle constant, producing an
    # optimal interception trajectory.
    # ----------------------------------------------------------
    accel_command = N * closing_vel * los_rate

    return accel_command