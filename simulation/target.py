import numpy as np
import config

class Target:

    def __init__(self):
        self.start = config.AIRCRAFT_START
        self.vel = config.TARGET_SPEED

    def position(self, t):

        if t <= config.STRAIGHT_TIME1:
            x = self.start[0] + self.vel * t
            y = self.start[1]
            z = self.start[2]

        elif t <= config.STRAIGHT_TIME1 + config.CURVE_TIME:

            tc = t - config.STRAIGHT_TIME1
            radius = (self.vel * config.CURVE_TIME) / config.TURN_ANGLE

            angle = tc * config.TURN_ANGLE / config.CURVE_TIME
            arc_angle = -np.pi/2 + angle

            x = radius * np.cos(arc_angle)
            y = radius * np.sin(arc_angle)
            z = self.start[2]

        else:

            ts = t - (config.STRAIGHT_TIME1 + config.CURVE_TIME)

            dx = np.cos(config.TURN_ANGLE)
            dy = np.sin(config.TURN_ANGLE)

            x = self.start[0] + self.vel * ts * dx
            y = self.start[1] + self.vel * ts * dy
            z = self.start[2]

        return np.array([x,y,z])
    
    """want to add
    random maneuvers
    AI evasive behavior
    """