import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(times, target_states, missile_states, result):
    """
    3D animated visualization of missile and target trajectories.

    Parameters
    ----------
    times : np.ndarray
        Time array from simulation
    target_states : np.ndarray (N, 3)
        Target position history
    missile_states : np.ndarray (N, 3)
        Missile position history
    result : SimResult
        Simulation result dataclass containing intercept info
    """

    # ----------------------------------------------------------
    # Find the frame index of intercept (if any) so we can
    # freeze both objects and show the marker from that point on
    # ----------------------------------------------------------
    intercept_frame = None
    if result.intercepted:
        intercept_time = result.intercept_time
        # Find the closest frame index to intercept time
        intercept_frame = int(np.argmin(np.abs(times - intercept_time)))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ----------------------------------------------------------
    # Set axis limits centered on both trajectories
    # ----------------------------------------------------------
    all_points = np.vstack([target_states, missile_states])
    padding = 0.1
    max_range = max(
        np.ptp(all_points[:, 0]),
        np.ptp(all_points[:, 1]),
        np.ptp(all_points[:, 2])
    )
    x_center = (all_points[:, 0].max() + all_points[:, 0].min()) / 2
    y_center = (all_points[:, 1].max() + all_points[:, 1].min()) / 2
    z_center = (all_points[:, 2].max() + all_points[:, 2].min()) / 2
    plot_radius = max_range / 2 * (1 + padding)

    ax.set_xlim(x_center - plot_radius, x_center + plot_radius)
    ax.set_ylim(y_center - plot_radius, y_center + plot_radius)
    ax.set_zlim(z_center - plot_radius, z_center + plot_radius)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Missile-Aircraft Pursuit Simulation — Proportional Navigation')
    ax.grid(True)
    ax.view_init(elev=20, azim=45)

    # ----------------------------------------------------------
    # Static markers for start positions
    # ----------------------------------------------------------
    ax.scatter(*target_states[0],  c='green',  s=100, marker='s', label='Aircraft Start')
    ax.scatter(*missile_states[0], c='orange', s=100, marker='^', label='Missile Start')

    # ----------------------------------------------------------
    # Intercept marker — initially hidden, shown only on intercept
    # ----------------------------------------------------------
    intercept_marker, = ax.plot([], [], [], 'r*', markersize=20, label='Intercept')

    # ----------------------------------------------------------
    # Animated artists
    # ----------------------------------------------------------
    target_point,  = ax.plot([], [], [], 'bo', markersize=10, label='Aircraft')
    target_trail,  = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5)
    missile_point, = ax.plot([], [], [], 'ro', markersize=8,  label='Missile')
    missile_trail, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.5)

    time_text     = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    distance_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
    status_text   = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10,
                              color='red', fontweight='bold')

    ax.legend()

    def init():
        for artist in (target_point, target_trail, missile_point, missile_trail,
                       intercept_marker):
            artist.set_data([], [])
            artist.set_3d_properties([])
        time_text.set_text('')
        distance_text.set_text('')
        status_text.set_text('')
        return (target_point, target_trail, missile_point, missile_trail,
                intercept_marker, time_text, distance_text, status_text)

    def update(frame):
        # --------------------------------------------------
        # If intercept has occurred, freeze both objects
        # at the intercept frame position
        # --------------------------------------------------
        if intercept_frame is not None and frame >= intercept_frame:
            frozen = intercept_frame
        else:
            frozen = frame

        # Aircraft — frozen at intercept if it occurred
        target_point.set_data([target_states[frozen, 0]], [target_states[frozen, 1]])
        target_point.set_3d_properties([target_states[frozen, 2]])
        target_trail.set_data(target_states[:frozen+1, 0], target_states[:frozen+1, 1])
        target_trail.set_3d_properties(target_states[:frozen+1, 2])

        # Missile — frozen at intercept if it occurred
        missile_point.set_data([missile_states[frozen, 0]], [missile_states[frozen, 1]])
        missile_point.set_3d_properties([missile_states[frozen, 2]])
        missile_trail.set_data(missile_states[:frozen+1, 0], missile_states[:frozen+1, 1])
        missile_trail.set_3d_properties(missile_states[:frozen+1, 2])

        # Intercept marker — only show from intercept frame onward
        if intercept_frame is not None and frame >= intercept_frame:
            ix, iy, iz = target_states[intercept_frame]
            intercept_marker.set_data([ix], [iy])
            intercept_marker.set_3d_properties([iz])
            status_text.set_text('INTERCEPT')
        else:
            intercept_marker.set_data([], [])
            intercept_marker.set_3d_properties([])
            status_text.set_text('')

        distance = np.linalg.norm(target_states[frozen] - missile_states[frozen])
        display_time = times[intercept_frame] if (intercept_frame is not None and frame >= intercept_frame) else times[frame]
        time_text.set_text(f'Time = {display_time:.2f} s')
        distance_text.set_text(f'Distance = {distance:.1f} m')

        return (target_point, target_trail, missile_point, missile_trail,
                intercept_marker, time_text, distance_text, status_text)

    frame_skip = max(1, len(times) // 500)
    frames = range(0, len(times), frame_skip)

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=False, interval=5, repeat=True)

    plt.show()