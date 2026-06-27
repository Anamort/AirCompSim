class SimulationBoundry(object):
    def __init__(self, x, y, z):
        self.maxX = x
        self.maxY = y
        self.maxZ = z

    def isInBoundry(self, x, y, z):
        if x <= self.maxX and y <= self.maxY and z <= self.maxZ and x >= 0 and y >= 0 and z >= 0:
            return True
        return False
