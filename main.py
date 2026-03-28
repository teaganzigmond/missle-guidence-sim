"""
Missile Guidance Simulation — Proportional Navigation
======================================================
Simulates a missile intercepting a maneuvering aircraft in 3D space.

Guidance law : Proportional Navigation (PN)
             Commands acceleration proportional to the LOS rotation rate,
             steering the missile onto a collision course rather than
             chasing the target's current position (pure pursuit).

Target path  : Straight → circular arc turn (3D) → straight

Tune the scenario in config.py — no changes needed here.
"""

from simulation.environment import Environment
from visualization.animation import animate


def main():
    # ── Run simulation ────────────────────────────────────────────────
    # Environment precomputes the target trajectory, then steps the
    # missile forward timestep-by-timestep using PN guidance.
    print("Running simulation...")
    env    = Environment()
    result = env.run()

    # ── Print summary ─────────────────────────────────────────────────
    # Shows intercept result, timing, miss distance, and key parameters.
    print(result.summary())

    # ── Animate ───────────────────────────────────────────────────────
    # Opens a 3D Matplotlib window with live position trails and HUD.
    print("Launching animation...")
    animate(result)


if __name__ == "__main__":
    main()