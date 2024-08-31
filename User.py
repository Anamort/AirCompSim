
from Location import Location
from Application import Task, Application
import numpy as np
import logging

class User(object):
    id: int = 0
    users = []

    def __init__(self, location: Location):
        self.id = User.id
        User.id += 1
        User.users.append(self)
        self.currentLocation: Location = location
        self.isMoving = False
        self.qoe: int = 0
        self.applications = set()
        self.city = ""
        self.trajectory = []

    @classmethod
    def removeUser(cls, userID):
        removedApps = []

        deletedUserIndex = -1
        for i, aUser in enumerate(User.users):
            if aUser.id == userID:
                deletedUserIndex = i
                break

        for application in User.users[deletedUserIndex].applications:
            removedApps.append(application.id)
        for appID in removedApps:
            Application.removeApplication(appID=appID)

        del User.users[deletedUserIndex]

        logging.info("User with id %s is removed.", str(userID))

    @classmethod
    def resetAll(cls):
        User.id = 0
        User.users = []

    def setApplication(self, app: Application):
        self.applications.add(app)
        app.userID = self.id

    def getQoE(self):
        return self.qoe

    def computeMovementDuration(self, loc: Location):
        distance = Location.getEuclideanDistance2D(loc1=self.currentLocation, loc2=loc)
        speed = 2  # currently it is fixed and 2 meter/sec ; this can be varied for each user
        return distance/speed

    def offload(self, task: Task):
        pass

    def getLocation(self):
        return self.currentLocation

    def getNearbyEdgeServer(self): # this will return a server
        pass



class OfficeUser(User):
    def __init__(self, lanID, location: Location):
        super().__init__(location)
        self.lanID: int = lanID


class MobileUser(User):

    def __init__(self, id, location):
        super().__init__(location)
        self.lanID: int = id
        self.speed = 0
        self.trajectory = [] # locations

    def getTrajectory(self):
        return self.trajectory

    def getNextLocation(self, radius):
        newLoc = Location(x=np.random.uniform(low=self.currentLocation.x - radius, high=self.currentLocation.x + radius),
                          y=np.random.uniform(low=self.currentLocation.y - radius, high=self.currentLocation.y + radius),
                          z=0)

        return newLoc


class FlyingUser(MobileUser):
    def __init__(self, altitude, id, location):
        super().__init__(id, location)
        self.altitude = altitude










