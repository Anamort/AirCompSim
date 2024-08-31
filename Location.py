import math
import numpy as np

class Location(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def getTerrestrialLocation(self):
        return [self.x, self.y]

    def getLocation(self):
        return [self.x, self.y, self.z]

    def __cmp__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        return False

    @classmethod
    def getEuclideanDistance2D(cls, loc1, loc2):
        return math.sqrt((abs(loc1.x - loc2.x)**2) + (abs(loc1.y - loc2.y)**2))

    @classmethod
    def getRandomLocWithin(cls, x, y):
        xLoc = np.random.uniform(low=0, high=x)
        yLoc = np.random.uniform(low=0, high=y)
        return Location(xLoc, yLoc, 200)

    @classmethod
    def getEuclideanDistance3D(self, loc1, loc2):
        return math.sqrt((abs(loc1.x - loc2.x) ** 2) + (abs(loc1.y - loc2.y) ** 2) + (abs(loc1.z - loc2.z) ** 2))






