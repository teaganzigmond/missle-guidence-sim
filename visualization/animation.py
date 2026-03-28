import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import config
from simulation.environment import SimResult


def animate(result: SimResult) -> None:
    """
    Display a 3D animated pursuit visualisation.

    Parameters
    ----------
    result : SimResult from Environment.run()
    """
    times   = result.times
    t_pos   = result.target_positions
    m_pos   = result.missile_positions
    dt      = config.DT

    # ── Figure & axes ────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # Fit both trajectories inside a cubic bounding box with padding.
    # Using a single max_range for all axes keeps the aspect ratio honest —
    # otherwise a trajectory that barely moves in Z looks stretched.
    all_pts   = np.vstack([t_pos, m_pos])
    padding   = 0.1
    max_range = max(np.ptp(all_pts[:, k]) for k in range(3))
    centres   = [(np.max(all_pts[:, k]) + np.min(all_pts[:, k])) / 2
                 for k in range(3)]
    r         = max_range / 2 * (1 + padding)   # half-width of each axis

    ax.set_xlim(centres[0] - r, centres[0] + r)
    ax.set_ylim(centres[1] - r, centres[1] + r)
    ax.set_zlim(centres[2] - r, centres[2] + r)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Missile–Aircraft Pursuit Simulation — Proportional Navigation')
    ax.grid(True)
    ax.view_init(elev=20, azim=45)

    # ── Static markers ───────────────────────────────────────────────
    ax.scatter(*t_pos[0],               c='green',  s=100, marker='s',
               label='Aircraft Start', zorder=5)
    ax.scatter(*config.MISSILE_START,   c='orange', s=100, marker='^',
               label='Missile Start',  zorder=5)

    if result.intercepted and result.intercept_index is not None:
        ax.scatter(*t_pos[result.intercept_index], c='red', s=250,
                   marker='*', label='Intercept', zorder=6)

    # ── Animated artists ─────────────────────────────────────────────
    tgt_dot,  = ax.plot([], [], [], 'bo', markersize=10, label='Aircraft')
    tgt_trail,= ax.plot([], [], [], 'b-', linewidth=2,   alpha=0.5,
                        label='Aircraft Trail')
    mis_dot,  = ax.plot([], [], [], 'ro', markersize=8,  label='Missile')
    mis_trail,= ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.5,
                        label='Missile Trail')

    time_txt  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    speed_txt = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
    dist_txt  = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10)
    state_txt = ax.text2D(0.02, 0.80, '', transform=ax.transAxes, fontsize=10,
                          color='red')

    ax.legend(loc='upper right')

    # ── Frame list (capped for performance) ──────────────────────────
    n_frames  = len(times)
    frame_skip= max(1, n_frames // config.MAX_ANIMATION_FRAMES)
    frames    = range(0, n_frames, frame_skip)

    # Pre-compute per-frame speed from true simulation indices
    # Use adjacent simulation frames so frame_skip doesn't distort readings
    speeds = np.empty(n_frames)
    speeds[0] = config.TARGET_SPEED
    for k in range(1, n_frames):
        speeds[k] = np.linalg.norm(t_pos[k] - t_pos[k - 1]) / dt

    # ── Init ─────────────────────────────────────────────────────────
    def init():
        """
        Reset all animated artists to empty.
        Called once by FuncAnimation before the first frame, and again
        on each loop repeat to clear the previous run's trails.
        """
        for artist in (tgt_dot, tgt_trail, mis_dot, mis_trail):
            artist.set_data([], [])
            artist.set_3d_properties([])
        for txt in (time_txt, speed_txt, dist_txt, state_txt):
            txt.set_text('')
        return tgt_dot, tgt_trail, mis_dot, mis_trail, time_txt, speed_txt, dist_txt, state_txt

    # ── Update ───────────────────────────────────────────────────────
    def update(frame):
        """
        Redraw all animated artists for the given simulation frame index.
        Trails are drawn by slicing positions up to the current frame,
        giving the growing-line effect without storing separate trail lists.
        """
        # Aircraft current position and cumulative trail
        tgt_dot.set_data([t_pos[frame, 0]], [t_pos[frame, 1]])
        tgt_dot.set_3d_properties([t_pos[frame, 2]])
        tgt_trail.set_data(t_pos[:frame + 1, 0], t_pos[:frame + 1, 1])
        tgt_trail.set_3d_properties(t_pos[:frame + 1, 2])

        # Missile current position and cumulative trail
        mis_dot.set_data([m_pos[frame, 0]], [m_pos[frame, 1]])
        mis_dot.set_3d_properties([m_pos[frame, 2]])
        mis_trail.set_data(m_pos[:frame + 1, 0], m_pos[:frame + 1, 1])
        mis_trail.set_3d_properties(m_pos[:frame + 1, 2])

        # HUD — distance is live missile-to-target separation
        dist = np.linalg.norm(t_pos[frame] - m_pos[frame])
        time_txt.set_text(f'Time     = {times[frame]:.2f} s')
        speed_txt.set_text(f'AC Speed = {speeds[frame]:.1f} m/s')
        dist_txt.set_text(f'Distance = {dist:.1f} m')

        # Show intercept banner from the intercept frame onwards
        if result.intercepted and result.intercept_index is not None \
                and frame >= result.intercept_index:
            state_txt.set_text('★ INTERCEPT')
        else:
            state_txt.set_text('')

        return tgt_dot, tgt_trail, mis_dot, mis_trail, time_txt, speed_txt, dist_txt, state_txt

    # ── Run ──────────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        blit=False, interval=config.ANIMATION_INTERVAL, repeat=True,
    )

    plt.tight_layout()
    plt.show()