import numpy as np
import config
from simulation.target import Target
from simulation.missile import Missile

class Environment:

    def __init__(self):

        self.target = Target()
        self.missile = Missile()

        self.times = np.arange(0, config.TMAX, config.DT)

        self.target_states = []
        self.missile_states = []

    def run(self):

        for t in self.times:

            target_pos = self.target.position(t)
            missile_pos = self.missile.step(target_pos, config.DT)

            self.target_states.append(target_pos)
            self.missile_states.append(missile_pos)

        self.target_states = np.array(self.target_states)
        self.missile_states = np.array(self.missile_states)

        return self.target_states, self.missile_states