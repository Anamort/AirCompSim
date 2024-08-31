import logging
from enum import Enum
from Location import Location

import numpy as np

class OffloadEntity(Enum):
    UserToEdge = 1
    UserToUAV = 2
    UserToCloud = 3
    EdgeToUAV = 4
    EdgeToCloud = 5
    UAVToCloud = 6
    UAVToEdge = 7


class Task(object):
    id: int = 0
    tasks = []
    def __init__(self, app, user, creationTime):
        from Server import Server
        from User import User
        self.id = Task.id
        Task.id += 1
        Task.tasks.append(self)
        self.creationTime = creationTime
        self.endTime: int = -1
        self.app = app
        self.user: User = user
        self.trajectory = [] # Locations
        self.processingTime = 0
        self.qos = -1 # between 0-100
        self.offloadEntity: OffloadEntity = None
        self.offloadLocation = user.getLocation()
        self.processLocation: Location = None
        self.processedServer: Server = None
        self.returnLocation: Location = None
        self.waitingTimeInQueue = 0
        self.isSuccess = False
        # TODO: OFFLOAD TIME

    @classmethod
    def resetAll(cls):
        Task.id = 0
        Task.tasks = []


    def getLatency(self):
        if self.endTime == -1:
            return -1
        return self.endTime - self.creationTime

    def getQoS(self):
        if self.getLatency() > self.app.worstDelay:
            self.qos = 0
        elif self.app.worstDelay > self.getLatency() and self.getLatency() < self.app.bestDelay:
            self.qos = 50
        else:
            self.qos = 100

    def isSuccessful(self):
        if self.getLatency() <= self.app.worstDelay:
            self.isSuccess = True


class ApplicationType(object):
    applicationTypes = {}
    def __init__(self, name, cpuCycle, worstDelay, bestDelay, exponential):
        self.name = name
        self.cpuCycle = cpuCycle
        self.worstDelay = worstDelay
        self.bestDelay = bestDelay
        self.interarrivalTime = exponential
        ApplicationType.applicationTypes[self.name] = [self.cpuCycle,
                                                       self.worstDelay,
                                                       self.bestDelay,
                                                       self.interarrivalTime]

    @classmethod
    def resetAll(cls):
        ApplicationType.applicationTypes = {}





class Application(object):
    id: int = 0
    applications = []
    def __init__(self, appType: ApplicationType, simTime):
        self.type = appType
        self.id = Application.id
        Application.id += 1
        Application.applications.append(self)
        self.tasks = []
        self.cpuCycle = appType.cpuCycle
        self.worstDelay = appType.worstDelay
        self.bestDelay = appType.bestDelay
        self.interarrivalTime = appType.interarrivalTime
        self.qos = 0
        self.waitingTime = 0
        self.innerTime = simTime
        self.userID = None

    @classmethod
    def removeApplication(cls, appID):
        deletedIndex = -1
        for i, app in enumerate(Application.applications):
            if app.id == appID:
                deletedIndex = i
                break
        del Application.applications[deletedIndex]
        logging.info("Edge server with id %s is removed.", str(appID))


    @classmethod
    def resetAll(cls):
        Application.id = 0
        Application.applications = []

    def generateTask(self, user) -> Task:
        # interarrivalTime = 1/lamda
        createdTaskTime = -np.log(1 - np.random.uniform(low=0, high=1)) * self.interarrivalTime
        #print("Task is created at: ", createdTaskTime)
        task = Task(app=self, user=user, creationTime=self.innerTime + createdTaskTime)
        logging.info("The task with id %s has been created at simulation time: %s for app %s", str(task.id), str(task.creationTime), str(self.id))
        task.offloadEntity = OffloadEntity.UserToEdge  # initially all tasks are produced by User
        self.tasks.append(task)
        self.innerTime += createdTaskTime
        return task

    def getMeanInterarrivalTime(self):  # TODO: Validated, it works correctly but must be updated and re-tested!!
        meanInterarrivalTime = 0
        if len(self.tasks) > 0:
            prev = self.tasks[0].creationTime
            for i in range(1, len(self.tasks)):
                meanInterarrivalTime = self.tasks[i].creationTime - prev
            meanInterarrivalTime = meanInterarrivalTime / len(self.tasks)
            logging.info("Mean interarrival for app %s is %s", str(self.id), str(meanInterarrivalTime))

    def isTaskValid(self, simTime):
        if self.innerTime < simTime:
            return True
        return False


    def getQoS(self):
        endedTaskCount = 0
        for task in self.tasks:
            if task.creationTime > 100:  # for warm-up period
                if task.qos != -1:
                    endedTaskCount += 1
                    self.qos += task.qos
                latency = task.getLatency()
                if latency > 0:
                    self.waitingTime += latency
        if endedTaskCount > 0: #for drl test cases
            self.qos = self.qos / endedTaskCount
            self.waitingTime = self.waitingTime / endedTaskCount
        logging.info("QoS of app %s is %s .", str(self.id), str(self.qos))
        logging.info("Avg waiting time of app %s is %s .", str(self.id), str(self.waitingTime))















