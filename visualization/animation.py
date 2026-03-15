import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate(times, target_states, missile_states):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    target_point, = ax.plot([],[],[], 'bo')
    missile_point, = ax.plot([],[],[], 'ro')

    def update(frame):

        target_point.set_data(
            [target_states[frame,0]],
            [target_states[frame,1]]
        )

        target_point.set_3d_properties(
            [target_states[frame,2]]
        )

        missile_point.set_data(
            [missile_states[frame,0]],
            [missile_states[frame,1]]
        )

        missile_point.set_3d_properties(
            [missile_states[frame,2]]
        )

        return target_point, missile_point

    anim = FuncAnimation(fig, update, frames=len(times))

    plt.show()