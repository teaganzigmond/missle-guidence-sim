"""
Missile Guidance Simulation

This project simulates missile interception of a maneuvering aircraft
using Proportional Navigation (PN), a widely used missile guidance law.

Proportional Navigation commands the missile to apply lateral
acceleration proportional to the rate of rotation of the line-of-sight
between the missile and target.

Typical navigation constants range from 3 to 5.
Ref: DTIC ADP010953

This simulation models:
    - missile kinematics under PN guidance
    - target motion (multi-segment trajectory)
    - lateral g-force limiting
    - interception logic
    - 3D trajectory visualization
"""

from simulation.environment import Environment
from visualization.animation import animate

env = Environment()
target_states, missile_states, result = env.run()

print("Launching animation...")
animate(env.times, target_states, missile_states, result)