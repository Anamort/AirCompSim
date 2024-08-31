import logging
from typing import List, Optional

from Location import Location
from Application import Task, Application
import math

class Server(object):

    def __init__(self, capacity: float,
                 location: Location,
                 radius: float,
                 power: float):
        self.capacity = capacity
        self.location: Location = location
        self.powerConsumption = power
        self.utilization = 0
        self.connectedComponents = []
        self.coverageBorders = []
        self.earliestIdleTime = 0  # TODO: will be removed
        self.nextAvailableTime = 0
        self.radius = radius
        self.innerTime = 0
        self.processedTasks = []  # for X seconds window



    def isInCoverage(self, loc: Location):
        if not self.location:
            print("Not self")
        if math.sqrt(pow(self.location.x - loc.x, 2) + pow(self.location.y - loc.y, 2)) <= self.radius:
            return True
        return False

    def getUtilization(self, timeLimit) -> float:
        if self.innerTime < timeLimit:
            self.innerTime = timeLimit

        curUtilization = self.utilization/self.innerTime

        if curUtilization > 1:
            print("UTILIZATION ERROR")
            exit(0)

        logging.info("OLD Utilization of server: %s percent", str(curUtilization*100))
        return curUtilization*100


    def getInstantUtilization(self, timeInterval, simTime):
        utilization = 0
        lengthOfTasks = len(self.processedTasks)
        maxTimeToGet = simTime - timeInterval

        if lengthOfTasks > 0:
            counter = lengthOfTasks-1
            tasks = []
            while counter > -1 \
                    and self.processedTasks[counter].creationTime + self.processedTasks[counter].waitingTimeInQueue  > maxTimeToGet \
                    and self.processedTasks[counter].creationTime + self.processedTasks[counter].waitingTimeInQueue < simTime:
                utilization += self.processedTasks[counter].processingTime
                tasks.append(self.processedTasks[counter])
                #if utilization > timeInterval:
                    #print("no")
                counter -= 1

            utilization = (utilization / timeInterval) * 100
            #utilization = math.floor(utilization)
            logging.info("Instant utilization of server: %s percent", str(utilization))
        else:
            return 0

        if math.floor(utilization) > 100:
            print("BUG")
            logging.info("UTILIZATION BUG")


        return utilization


    def getConnectedComponents(self):
        return self.connectedComponents

    def getNumberOfConnectedComponents(self):
        return len(self.connectedComponents)

    #TODO:
    def getEnergyConsumption(self):
        pass


    def getProcessingDelay(self, task: Task):
        app: Application = task.app
        mu = self.capacity/app.cpuCycle # 1/seconds
        # totalDelay = 1/(mu - (1/app.interarrivalTime))
        task.processingTime = 1/mu
        self.utilization += task.processingTime

        self.nextAvailableTime += task.processingTime
        self.innerTime += task.processingTime
        # self.earliestIdleTime += task.processingTime
        return task.processingTime

    def updateEarliestIdleTime(self, val):
        self.earliestIdleTime -= val

    def updateProcessedTasks(self, task: Task):
        self.processedTasks.append(task)





class EdgeServer(Server):
    edgeServers = []
    id: int = 0

    @classmethod
    def resetAll(cls):
        EdgeServer.edgeServers = []
        EdgeServer.id = 0

    @classmethod
    def removeEdgeServer(cls, id):
        del EdgeServer.edgeServers[id]
        logging.info("Edge server with id %s is removed.", str(id))

    def __init__(self, capacity: float, location: Location, radius: float, power: float):
        super().__init__(capacity, location, radius, power)
        self.id = EdgeServer.id
        EdgeServer.edgeServers.append(self)
        EdgeServer.id += 1

    def __eq__(self, other):
        return self.id == other.id


class UAV(Server):
    uavs = []
    activeUAVs = []
    reserveUAVs = []
    id: int = 0

    @classmethod
    def resetAll(cls):
        UAV.uavs = []
        UAV.id = 0

    def __init__(self, location: Location, capacity: float, radius: float, power: float):
        super().__init__(capacity, location, radius, power)
        self.id = UAV.id
        UAV.id += 1
        self.location = location
        self.batteryLevel = 100
        self.horizontalSpeed = 0
        self.verticalSpeed = 0
        self.altitude = 200
        self.trajectory = []
        self.trajectory.append(location)
        self.isFlying = False
        self.notFlyingSince = 0
        self.flyingTo = None
        UAV.uavs.append(self)

    def __eq__(self, other):
        return self.id == other.id

    def getSpeed(self):
        return math.sqrt((self.horizontalSpeed ** 2) + (self.verticalSpeed ** 2))



    def computeFlightDurationBasedOn(self, loc: Location):
        distance = Location.getEuclideanDistance2D(loc1=self.location, loc2=loc)
        speed = 2.5  # currently it is fixed and 10 meter/sec
        return distance/speed


    def getHorizontalSpeed(self):
        return self.horizontalSpeed

    def getVerticalSpeed(self):
        return self.verticalSpeed

    #TODO
    def updateHorizontalSpeed(self):
        pass
    #TODO
    def updateVerticalSpeed(self):
        pass


    def getBatteryLevel(self):
        return self.batteryLevel

    #TODO
    def updateBatteryLevel(self):
        pass


    #TODO
    def getTransmissionCapacity(self, altitude):
        pass

    #TODO
    def getCoveredUsersInGround(self):
        pass

    def getTrajectory(self):
        return self.trajectory

    #TODO
    def updateTrajectory(self):
        pass


class CloudServer(Server):
    def __init__(self, capacity: float, location: Location, radius: float, power: float):
        super().__init__(capacity, location, radius, power)
