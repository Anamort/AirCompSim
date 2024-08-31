

# from Server import Server, EdgeServer, UAV
# from Event import Event
import heapq
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from User import User
from Application import Application, Task, OffloadEntity, ApplicationType
from Location import Location
from Server import Server, EdgeServer, UAV, CloudServer
from Event import Event, EventType
from Scenario import Scenario
from DRL import ActorCriticAgent, ActorCriticNetwork, MemoryItem, State, Trainer
from DQN import DQNAgent
from DDQN import DDQNAgent
import random


# Simulation Boundries are get from this class
class SimulationBoundry(object):
    def __init__(self, x, y, z):
        self.maxX = x
        self.maxY = y
        self.maxZ = z

    def isInBoundry(self, x, y, z):
        if x <= self.maxX and y <= self.maxY and z <= self.maxZ and x >= 0 and y >= 0 and z >= 0:
            return True
        return False


class Simulation(object):
    def __init__(self, userCount, edgeCount, uavCount, testNumber, flyPolicy, waitingPolicy, userMobilityPolicy, agent, uavRadius, seedNo, isDRLTraining, locations):
        self.simulationTime = 0
        self.eventQueue = []  # heap of events
        self.boundry = SimulationBoundry(400, 400, 400)
        self.numberOfUsers = userCount
        self.numberOfServers = edgeCount
        self.numberOfUAVs = uavCount
        self.testNumber = testNumber
        self.uavFlyPolicy = flyPolicy
        self.uavWaitingPolicy = waitingPolicy
        self.userMobilityPolicy = userMobilityPolicy
        self.lastFlyingPolicyApplicationTime = 0
        self.isUAVMethodApplied = False  # for initial usage of UAV methods regardless of uavWaitingPolicy
        self.isUAVDebug = False



        # Essential for uav clusters!
        self.isUAVHeuristic = False


        self.agent = agent
        self.reward = 0
        self.state = 0
        self.isState = False
        self.isDRLDone = False
        self.score = 0
        self.isDRLTraining = isDRLTraining
        self.locations = locations

        self.uavRadius = uavRadius
        self.seedNo = seedNo


        self.successTaskCountForEpisode = 0



        Application.resetAll()
        User.resetAll()
        EdgeServer.resetAll()
        UAV.resetAll()
        Task.resetAll()
        Event.resetAll()
        ApplicationType.resetAll()



    def StartSimulation(self):
        simulationTime: float = 0.1
        timeLimit: float = 1000
        timeStepForUsers = 2
        stateInterval = timeStepForUsers + 1  # stateInterval is used for consecutive states for DRL


        cloudLocation = Location(x=self.boundry.maxX, y=self.boundry.maxY, z=0)
        cloudServer = CloudServer(capacity=100000, location=cloudLocation, radius=self.boundry.maxX, power=100)

        simScenario = Scenario(numberOfUAVs=self.numberOfUAVs,
                               numberOfUsers=self.numberOfUsers,
                               UAVFlyPolicy=self.uavFlyPolicy,
                               UAVWaitingPolicy=self.uavWaitingPolicy,
                               testNo=self.testNumber,
                               uavRadius=self.uavRadius,
                               scenario="BasicEdge")
        simScenario.basicEdgeScenario()





        if self.isDRLTraining:
            simScenario.isDRL = True
            simScenario.DRLScenario(seedNumber=self.seedNo) # seedNo can be ignored based on research
        else:
            self.isUAVHeuristic = True
            # if not self.locations:
            #    raise Exception("Locations should be learned before heuristic uav approaches!!")
            # sys.exit(1)
            simScenario.isDRL = False
            for edge in EdgeServer.edgeServers:
                # edge locations are used for the existing UAV policy
                # this behavior can be changed by researchers based their experiments
                self.locations.append((edge.location.x, edge.location.y))


        # Test scenarios

        # simScenario.mobilityTestScenario()

        # simScenario.uavTestScenario()



        if simScenario.isDRL:
            #  If DRL is used, the corresponding states based on the stateInterval are added to the event mechanism
            for stateTime in range(1, int(timeLimit), stateInterval):
                stateEvent: Event = Event(type=EventType.State, task=None, simTime=stateTime, user=None, uav=None, loc=None)
                heapq.heappush(self.eventQueue, stateEvent)


        from Mobility import Mobility

        while simulationTime < timeLimit:

            # This is for the event-based dynamic scenarios
            simScenario.updateScenario(simulationTime, heapq, self.eventQueue)

            for user in User.users:
                for app in user.applications:
                    if app.isTaskValid(simulationTime):
                        newTask: Task = app.generateTask(user)
                        if newTask.creationTime < timeLimit:
                            # User field can be empty here since task also includes the user info
                            # Each task is added to the event mechanism to be processed
                            newEvent: Event = Event(type=EventType.Offload, task=newTask,
                                                    simTime=newTask.creationTime, user=None, uav=None, loc=user.currentLocation)
                            heapq.heappush(self.eventQueue, newEvent)


                if simScenario.userMobility and not user.isMoving:
                    user.trajectory.append(user.currentLocation)
                    newLoc = Mobility.moveUser(currentLoc=user.currentLocation, simBoundry=self.boundry, radius=350)
                    movementDuration = user.computeMovementDuration(loc=newLoc)
                    newEvent: Event = Event(type=EventType.UserStop, task=None, simTime=simulationTime + movementDuration , user=user, uav=None, loc=newLoc)
                    logging.info("User %s has started to move from location %s to location %s with duration of %s seconds ",
                                 str(user.id), user.currentLocation, newLoc, str(movementDuration))
                    user.isMoving = True
                    heapq.heappush(self.eventQueue, newEvent)
                    slope = (newLoc.y - user.currentLocation.y) / (newLoc.x - user.currentLocation.x)
                    slope = math.fabs(slope)
                    isXNeg = False
                    isYNeg = False
                    if newLoc.y - user.currentLocation.y < 0:
                        isYNeg = True
                    if newLoc.x - user.currentLocation.x < 0:
                        isXNeg = True
                    angle = np.arctan(slope)
                    distance = 2 * timeStepForUsers  # TODO: Each user should have its own speed since currently they move 2 m/s as fixed
                    prevLocationPoint: Location = Location(x=user.currentLocation.x, y=user.currentLocation.y, z=0)

                    # Each step of each user is considered and therefore cause an event in the simulation
                    for stepTime in range(int(math.ceil(simulationTime)) + timeStepForUsers, int(math.floor(simulationTime + movementDuration)), timeStepForUsers):
                        nextX = 0.0
                        nextY = 0.0
                        if isXNeg:
                            nextX = prevLocationPoint.x - (distance * np.cos(angle))
                        else:
                            nextX = (distance * np.cos(angle)) + prevLocationPoint.x
                        if isYNeg:
                            nextY = prevLocationPoint.y - (distance * np.sin(angle))
                        else:
                            nextY = prevLocationPoint.y + (distance * np.sin(angle))
                        nextLocationPoint: Location = Location(x=nextX, y=nextY, z=0)
                        prevLocationPoint.x = nextLocationPoint.x
                        prevLocationPoint.y = nextLocationPoint.y
                        newEvent: Event = Event(type=EventType.UserMove, task=None,
                                                simTime=stepTime, user=user, uav=None, loc=nextLocationPoint)
                        heapq.heappush(self.eventQueue, newEvent)



            # START UAV POLICY

            '''
            Current UAV policies such as LSI and Random relies on given edge locations. Therefore UAVs move to those
            locations to enhance the existing capacity of the location. Additional methods, independent from edge 
            locations and therefore based on user-concentrated areas, will be developed later.
            '''

            if self.isUAVHeuristic and self.uavFlyPolicy != "NoUAV" and len(UAV.uavs) > 0:
                if not self.isUAVMethodApplied or (simulationTime - self.lastFlyingPolicyApplicationTime > self.uavWaitingPolicy):
                    clusterLocations = None
                    if self.uavFlyPolicy == "TaskCluster": # This is an experimental policy
                        clusterLocations = Mobility.newTaskBasedClustering(numberOfClusters=len(self.locations))  # This is an experimental method
                    elif self.uavFlyPolicy == "LSI":
                        clusterLocations = Mobility.locationSelectionIndex(numberOfLocations=len(self.locations), locations=self.locations, uavRadius=simScenario.uavRadius)  # TODO: numberOfClusters must be based on scenario
                    if (self.uavFlyPolicy != "Random" and len(clusterLocations) > 0) or self.uavFlyPolicy == "Random":
                        clusterLocations = Mobility.sortUAVsForClusterLocations(clusterLocations) # min distance is selected for UAV-clusterLoc pairs in this function
                        clusterNo = 0
                        for uav in UAV.uavs:
                            newLoc: Location = None
                            if self.uavFlyPolicy == "Random":
                                if not uav.isFlying and simulationTime - uav.notFlyingSince > self.uavWaitingPolicy:
                                    newLoc: Location = Mobility.randomUAVMove(locations=self.locations)
                                else:
                                    continue
                            else:
                                newLoc: Location = clusterLocations[clusterNo]


                            travelTime = uav.computeFlightDurationBasedOn(loc=newLoc)
                            newEvent: Event = Event(type=EventType.UAVStop, task=None,
                                                    simTime=simulationTime + travelTime,
                                                    user=None, uav=uav, loc=newLoc)
                            logging.info(
                                "%s: Uav take off from location %s to %s. The travel time: %s. The arrival will be %s",
                                self.uavFlyPolicy, uav.location, newLoc, str(travelTime), str(simulationTime + travelTime))
                            heapq.heappush(self.eventQueue, newEvent)
                            uav.isFlying = True
                            uav.flyingTo = newLoc
                            clusterNo += 1

                            if self.uavFlyPolicy == "LSI" and clusterNo == len(clusterLocations):
                                break


                        self.lastFlyingPolicyApplicationTime = simulationTime
                        self.isUAVMethodApplied = True


                        if self.isUAVDebug:
                            locXUsers = []
                            locYUsers = []
                            for user in User.users:
                                locXUsers.append(user.getLocation().x)
                                locYUsers.append(user.getLocation().y)

                            uavLocationsX = []
                            uavLocationsY = []
                            for cLoc in clusterLocations:
                                uavLocationsX.append(cLoc.x)
                                uavLocationsY.append(cLoc.y)

                            plt.figure()
                            plt.scatter(locXUsers, locYUsers)
                            plt.scatter(uavLocationsX, uavLocationsY, color="red")  # , alpha=0.2, markersize=50
                            for uavX, uavY in zip(uavLocationsX, uavLocationsY):
                                circle1 = plt.Circle((uavX, uavY), 100, color='red', alpha=0.2)
                                plt.gca().add_patch(circle1)

                            plt.savefig("UAV-and-"+self.uavFlyPolicy+".pdf")

            # END UAV POLICY

            elif simScenario.uavMobilityTest:
                for uav in UAV.uavs:
                    if not uav.isFlying:
                        newLoc = Mobility.moveUser(currentLoc=uav.location, simBoundry=self.boundry, radius=500)
                        travelTime = uav.computeFlightDurationBasedOn(loc=newLoc)
                        newEvent: Event = Event(type=EventType.UAVStop, task=None,
                                                simTime=simulationTime + travelTime,
                                                user=None, uav=uav, loc=newLoc)
                        logging.info(
                            "%s: Uav take off from location %s to %s. The travel time: %s. The arrival will be %s",
                            self.uavFlyPolicy, uav.location, newLoc, str(travelTime), str(simulationTime + travelTime))
                        heapq.heappush(self.eventQueue, newEvent)
                        uav.isFlying = True
                        uav.flyingTo = newLoc

                        # START STEPS of UAV

                        slope = (newLoc.y - uav.location.y) / (newLoc.x - uav.location.x)
                        slope = math.fabs(slope)
                        isXNeg = False
                        isYNeg = False
                        if newLoc.y - uav.location.y < 0:
                            isYNeg = True
                        if newLoc.x - uav.location.x < 0:
                            isXNeg = True
                        angle = np.arctan(slope)

                        distance = 10 * timeStepForUsers  # TODO: each uav has its own speed
                        prevLocationPoint: Location = Location(x=uav.location.x, y=uav.location.y, z=0)

                        for stepTime in range(int(math.ceil(simulationTime)) + timeStepForUsers,
                                              int(math.floor(simulationTime + travelTime)), timeStepForUsers):
                            nextX = 0.0
                            nextY = 0.0
                            if isXNeg:
                                nextX = prevLocationPoint.x - (distance * np.cos(angle))
                            else:
                                nextX = (distance * np.cos(angle)) + prevLocationPoint.x
                            if isYNeg:
                                nextY = prevLocationPoint.y - (distance * np.sin(angle))
                            else:
                                nextY = prevLocationPoint.y + (distance * np.sin(angle))
                            nextLocationPoint: Location = Location(x=nextX, y=nextY, z=0)
                            prevLocationPoint.x = nextLocationPoint.x
                            prevLocationPoint.y = nextLocationPoint.y

                            newEvent: Event = Event(type=EventType.UAVMove, task=None, simTime=stepTime,
                                                    user=None,
                                                    uav=uav, loc=nextLocationPoint)

                            heapq.heappush(self.eventQueue, newEvent)




            if len(self.eventQueue) > 0:
                event: Event = heapq.heappop(self.eventQueue)
                simulationTime = event.scheduledTime
                logging.info("SimTime: %s --> New event with EventType %s is popped from the heap.", str(simulationTime), str(event.type))


                if event.type == EventType.Offload:
                    logging.info("SimTime: %s --> Offloading task id %s --> App id of the task is %s ---> User id of the task is %s ."
                                 , str(simulationTime), str(event.task.id), str(event.task.app.id), str(event.task.user.id))
                    theTask: Task = event.task

                    if theTask.offloadEntity == OffloadEntity.UserToEdge: #TODO: UserToEdge can be changed
                        # 1) Check if there is an available edge server
                        # 2) Otherwise check UAV availability and send it
                        # 3) Otherwise send it to the cloud
                        logging.info("SimTime: %s --> Offloaded user location: %s", str(simulationTime), theTask.user.getLocation())
                        theUser: User = theTask.user

                        availableEdgeServers = []
                        for edgeServer in EdgeServer.edgeServers:
                            if edgeServer.isInCoverage(theUser.getLocation()):
                                logging.info("SimTime: %s ---> Edge server %s, at location %s, covers the user %s",
                                             str(simulationTime), str(edgeServer.id), edgeServer.location, str(theUser.id))
                                availableEdgeServers.append(edgeServer)

                        availableUAVs = []
                        for uav in UAV.uavs:
                            logging.info("For uav %s is in coverage check", str(uav.id))

                            if not uav.isFlying and uav.isInCoverage(theUser.getLocation()): # and not uav.isFlying: # FOR EARTHQUAKE THIS WAS ACTIVE!!
                                logging.info("SimTime: %s ---> UAV %s, at location %s, can be used for the user %s",
                                             str(simulationTime), str(uav.id), uav.location,
                                             str(theUser.id))
                                availableUAVs.append(uav)

                        theEdgeServer: EdgeServer = None
                        theUAV: UAV = None
                        isUAV = False
                        isEdge = False
                        if len(availableEdgeServers) > 0:
                            theEdgeServer = availableEdgeServers[0]

                            for edge in availableEdgeServers:
                                if edge.nextAvailableTime < theEdgeServer.nextAvailableTime:
                                    theEdgeServer = edge


                        if len(availableUAVs) > 0:
                            theUAV = availableUAVs[0]

                            for uav in availableUAVs:
                                if uav.nextAvailableTime < theUAV.nextAvailableTime:
                                    theUAV = uav

                        if theUAV and theEdgeServer:

                            '''
                            Currently the server type whose next available time is the closest is selected for offloading.
                            However, this behevior can be changed based on the research.  
                            '''

                            if theUAV.nextAvailableTime < theEdgeServer.nextAvailableTime:
                                isUAV = True

                            else:
                                isEdge = True
                        elif theUAV:
                            isUAV = True
                        elif theEdgeServer:
                            isEdge = True

                        newProcessEvent: Event = None



                        if isEdge:
                            theTask.processedServer = theEdgeServer

                            if theEdgeServer.nextAvailableTime < simulationTime:
                                theEdgeServer.nextAvailableTime = simulationTime  # it means now it is available
                                theTask.waitingTimeInQueue = 0
                            else:
                                theTask.waitingTimeInQueue = theEdgeServer.nextAvailableTime - simulationTime  # TODO: Network delay should be considered later
                            nextAvailableTime = theEdgeServer.nextAvailableTime
                            newProcessEvent = Event(type=EventType.Process, task=theTask,
                                                    simTime=nextAvailableTime + theEdgeServer.getProcessingDelay(
                                                        theTask), user=None, uav=None, loc=theEdgeServer.location)
                            logging.info("SimTime: %s ---> For the task %s, the edge server %s is selected for offloading.",
                                         str(simulationTime), str(theTask.id), str(theEdgeServer.id))
                        elif isUAV:
                            theTask.processedServer = theUAV
                            if theUAV.nextAvailableTime < simulationTime:
                                theUAV.nextAvailableTime = simulationTime  # it means now it is available
                                theTask.waitingTimeInQueue = 0
                            else:
                                theTask.waitingTimeInQueue = theUAV.nextAvailableTime - simulationTime  # TODO: Network delay should be considered later
                            nextAvailableTime = theUAV.nextAvailableTime


                            newProcessEvent = Event(type=EventType.Process, task=theTask,
                                                    simTime=nextAvailableTime + theUAV.getProcessingDelay(
                                                        theTask), user=None, uav=None, loc=theUAV.location)

                            logging.info(
                                "SimTime: %s ---> For the task %s, the UAV %s is selected for offloading.",
                                str(simulationTime), str(theTask.id), str(theUAV.id))
                        else:  # to the cloud
                            theTask.processedServer = cloudServer
                            theTask.waitingTimeInQueue = 0  # we assume that cloud doesn't cause any queueing delay
                            newProcessEvent = Event(type=EventType.Process, task=theTask,
                                                    simTime=simulationTime + cloudServer.getProcessingDelay(theTask), user=None, uav=None, loc=cloudServer.location)
                            logging.info(
                                "SimTime: %s ---> For the task %s, the cloud is selected for offloading.",
                                str(simulationTime), str(theTask.id))

                        heapq.heappush(self.eventQueue, newProcessEvent)



                elif event.type == EventType.Process:  # it means that processing is OVER!
                    processedTask = event.task
                    processedTask.processedServer.updateProcessedTasks(processedTask)
                    user = processedTask.user

                    newReturnedEvent: Event = None
                    # TODO: Currently the network delay is fixed. It should be dynamic based on the size of tasks and the
                    #  capacity of the network. Therefore, a network model will be developed in the next version of AirCompSim!

                    if isinstance(processedTask.processedServer, EdgeServer):
                        newReturnedEvent = Event(EventType.Returned, processedTask, simulationTime + 0.001, uav=None, user=user, loc=user.currentLocation)
                    elif isinstance(processedTask.processedServer, UAV):
                        newReturnedEvent = Event(EventType.Returned, processedTask, simulationTime + 0.005, uav=None, user=user, loc=user.currentLocation)
                    else:  # cloud
                        newReturnedEvent = Event(EventType.Returned, processedTask, simulationTime + 1.5, uav=None, user=user, loc=user.currentLocation)

                    heapq.heappush(self.eventQueue, newReturnedEvent)
                    logging.info("SimTime: %s ---> The task with id %s is processed.",
                                 str(simulationTime), str(event.task.id))

                elif event.type == EventType.Returned:
                    logging.info("SimTime: %s ---> The task with id %s has been returned and completed.",  str(simulationTime), str(event.task.id))
                    event.task.endTime = simulationTime
                    event.task.getQoS()
                    event.task.isSuccessful()
                    if simScenario.isDRL:
                        if event.task.isSuccess:
                            #self.reward += 100
                            self.successTaskCountForEpisode += 1

                elif event.type == EventType.UAVStop:
                    uav = event.uav
                    logging.info("SimTime: %s ---> UAV %s has arrived its destination now at location %s", str(simulationTime), uav.id, uav.flyingTo)
                    uav.notFlyingSince = simulationTime
                    dst = uav.flyingTo
                    uav.location = dst
                    uav.trajectory.append(uav.location)
                    #uav.flyingTo = None
                    uav.isFlying = False

                elif event.type == EventType.UAVMove:
                    uav = event.uav
                    newLocation = event.location
                    logging.info("SimTime: %s ---> UAV %s has been moved and now at location %s", str(simulationTime), uav.id, newLocation)
                    uav.location = newLocation
                    uav.trajectory.append(uav.location)

                elif event.type == EventType.UserMove:
                    user = event.user
                    prevLocPoint = user.currentLocation
                    logging.info("SimTime: %s ---> User %s at location %s has been moved as stepTime and now at location %s", str(simulationTime), user.id, prevLocPoint, event.location)
                    user.currentLocation = event.location
                    user.trajectory.append(user.currentLocation)

                elif event.type == EventType.UserStop:
                    user = event.user
                    prevLocPoint = user.currentLocation
                    logging.info(
                        "SimTime: %s ---> User %s at location %s has been moved and stop now at location %s",
                        str(simulationTime), user.id, prevLocPoint, event.location)
                    user.currentLocation = event.location
                    user.trajectory.append(user.currentLocation)
                    user.isMoving = False

                elif event.type == EventType.State:
                    '''
                    This part is left to developers/researches since DRL can be used in many different ways.
                    However, we provide an skeleton code in order to show how the corresponding actions taken by 
                    multiple UAVs and then added to the event mechanism.
                    
                    '''
                    logging.info("SimTime: %s ---> State is computed", str(simulationTime))
                    # 1) Get current state, which is the next state of prevState
                    # 2) Compute the reward
                    # 3) Considering prevState, prevAction, reward, prevAction, and isDone, create a MemoryItem
                    # 4) Get 64 (batch size) random memoryItem and then train the agent
                    # 5) Make an action for the current state

                    if self.isDRLDone:
                        logging.info("SimTime: %s ---> DRL has been already completed. No need for training for this episode!", str(simulationTime))

                    else:
                        currentState = State(simTime=event.scheduledTime).getState()
                        action = []

                        if self.isDRLTraining:
                            '''
                                Each UAV can be considered as an DRL agent. 
                            '''
                            for anAgent in self.agent:
                                action.append(anAgent.action(currentState))

                        else:
                            '''
                               Each UAV can be considered as an DRL agent. 
                            '''

                            for anAgent in self.agent:
                                action = anAgent.predictAction(currentState)

                        '''
                            In here, we coded an discrete action space for each agent (UAV). Therefore actions are
                            left, right, up, down, and notMove. Each agent/UAV can move as "distance" variable between 
                            two consecutive states.
                        '''

                        distance = timeStepForUsers * 2.5


                        for i, uav in enumerate(UAV.uavs):
                            anAgent = self.agent[i]
                            newX = uav.location.x
                            newY = uav.location.y
                            if action[i] == 0:
                                # noMove
                                #print("noMove")
                                logging.info("%s: No move for UAV %s", self.uavFlyPolicy, str(uav.id))
                            elif action[i] == 1:
                                # left
                                newX -= distance
                            elif action[i] == 2:
                                # up
                                newY += distance
                            elif action[i] == 3:
                                # right
                                newX += distance
                            elif action[i] == 4:
                                newY -= distance
                            if newX < 0 or newY < 0 or newX > self.boundry.maxX or newY > self.boundry.maxY:
                                anAgent.reward -= 1
                            newX = min(self.boundry.maxX, newX)
                            newX = max(0, newX)
                            newY = min(self.boundry.maxY, newY)
                            newY = max(0, newY)

                            '''
                                Based on the action, new state is computed and then added to the event queue.
                            '''

                            newLoc = Location(x=newX, y=newY, z=uav.location.z)
                            travelTime = uav.computeFlightDurationBasedOn(loc=newLoc)
                            newEvent: Event = Event(type=EventType.UAVStop, task=None,
                                                    simTime=simulationTime + travelTime,
                                                    user=None, uav=uav, loc=newLoc)
                            logging.info(
                                "%s: Uav %s take off from location %s to %s. The travel time: %s. The arrival will be %s",
                                self.uavFlyPolicy, str(uav.id), uav.location, newLoc, str(travelTime),
                                str(simulationTime + travelTime))

                            heapq.heappush(self.eventQueue, newEvent)
                            uav.isFlying = True
                            uav.flyingTo = newLoc

                            '''
                                Reward of an agent can be computed as follows. Reward mechanism can be different based
                                on the research.
                            '''

                            for aUser in User.users:
                                dist = (math.sqrt(pow(newLoc.x - aUser.currentLocation.x, 2) + pow(newLoc.y - aUser.currentLocation.y, 2)))
                                point = 0
                                if dist != 0:
                                    point = 1 / (math.sqrt(pow(newLoc.x - aUser.currentLocation.x, 2) + pow(newLoc.y - aUser.currentLocation.y, 2)))
                                else:
                                    point = 1
                                # if aUser.id not in userSet:
                                #     anAgent.reward += point
                                anAgent.reward += point
                                if math.sqrt(pow(newLoc.x - aUser.currentLocation.x, 2) + pow(newLoc.y - aUser.currentLocation.y, 2)) <= uav.radius:
                                    anAgent.reward += 1


                        if self.isDRLTraining:

                            if self.isState:
                                index = 0
                                for anAgent in agent: # considering multiple agents
                                    MemoryItem(state=self.state,
                                               nextState=currentState,
                                               reward=anAgent.reward,
                                               action=action[index],
                                               isDone=self.isDRLDone)  # stored inside the class
                                    anAgent.memoryItems.append((self.state, action[index], anAgent.reward, currentState))
                                    self.score += anAgent.reward  # self.successTaskCountForEpisode
                                    anAgent.learn()
                                    anAgent.reward = 0
                                    index += 1

                            else:
                                self.isState = True
                            self.state = currentState
                        else:
                            self.score += self.reward

                else:
                    print("Impossible!!")
            else:
                logging.info("No further event!!! Simulation time: %s , Time Limit: %s", str(simulationTime),
                             str(timeLimit))
                break





        # pandas dataframes
        appResults = {"TestNo": self.testNumber,
                      "NumberOfEdges": self.numberOfServers,
                      "NumberOfUAVs": self.numberOfUAVs,
                      "NumberOfUsers": len(User.users),
                      "UAVFlyPolicy": self.uavFlyPolicy,
                      "UAVWaitingPolicy": self.uavWaitingPolicy,
                      "UserMobilityPolicy": self.userMobilityPolicy,
                      "QueueingDelays": [],  # for each application
                      "ApplicationTypes": [],
                      "TotalTasks": [],  # for each application (type)
                      "SuccessfulTasks": [],  # for each application (type)
                      "OffloadedToUAV": [], # for each application (type)
                      "OffloadedToEdge": [],  # for each application (type)
                      "OffloadedToCloud": [],  # for each application (type)
                      "Users": [],  # for each user (id)
                      "DRLScore": self.score,
                      "UAVRadius": self.uavRadius
                      }

        edgeResults = {"TestNo": self.testNumber,
                       "NumberOfEdges": self.numberOfServers,
                       "NumberOfUAVs": self.numberOfUAVs,
                       "NumberOfUsers": self.numberOfUsers,
                       "UAVFlyPolicy": self.uavFlyPolicy,
                       "UAVWaitingPolicy": self.uavWaitingPolicy,
                       "EdgeUtilization": []  # for each edge server
                       }

        uavResults = {"TestNo": self.testNumber,
                      "NumberOfEdges": self.numberOfServers,
                      "NumberOfUAVs": self.numberOfUAVs,
                      "NumberOfUsers": self.numberOfUsers,
                      "UAVFlyPolicy": self.uavFlyPolicy,
                      "UAVWaitingPolicy": self.uavWaitingPolicy,
                      "UAVUtilization": [],  # for each uav
                      "Trajectory": [] #trajectory for each uav
                      }


        for anApp in Application.applications:
            totalTasks = 0 #len(anApp.tasks)
            successfulTasks = 0
            anApp.getMeanInterarrivalTime()
            anApp.getQoS()
            queueingDelay = 0
            offloadedToUav = 0
            offloadedToEdge = 0
            offloadedToCloud = 0
            for aTask in anApp.tasks:
                if aTask.creationTime > 100:  # after warm-up period
                    totalTasks += 1
                    queueingDelay += aTask.waitingTimeInQueue
                    if aTask.isSuccess:
                        successfulTasks += 1
                    if isinstance(aTask.processedServer, EdgeServer):
                        offloadedToEdge += 1
                    elif isinstance(aTask.processedServer, UAV):
                        offloadedToUav += 1
                    else:
                        offloadedToCloud += 1
            appResults["TotalTasks"].append(totalTasks)
            appResults["SuccessfulTasks"].append(successfulTasks)
            appResults["OffloadedToEdge"].append(offloadedToEdge)
            appResults["OffloadedToUAV"].append(offloadedToUav)
            appResults["OffloadedToCloud"].append(offloadedToCloud)
            appResults["Users"].append(anApp.userID)
            appResults["ApplicationTypes"].append(anApp.type.name)

            if len(anApp.tasks) > 0:
                queueingDelay = queueingDelay / (len(anApp.tasks))
            appResults["QueueingDelays"].append(queueingDelay)



        for anEdgeServer in EdgeServer.edgeServers:
            utilization = anEdgeServer.getUtilization(timeLimit)
            edgeResults["EdgeUtilization"].append(utilization)

        for uav in UAV.uavs:
            utilization = uav.getUtilization(timeLimit)
            uavResults["UAVUtilization"].append(utilization)
            uavResults["Trajectory"].append(utilization)

        appResults = pd.DataFrame(appResults)
        edgeResults = pd.DataFrame(edgeResults)
        uavResults = pd.DataFrame(uavResults)

        scenarioResults = simScenario.computeMetrics()




        return [appResults, edgeResults, uavResults, scenarioResults]



if __name__ == '__main__':
    '''
        AirCompSim can be run from here. You can add parsers as follows, however the existing version get those information
        from lists. Even though lists such as numberOfUsers, numberOfUAVs is set from here, Scenario module can also be used to
        for this purpose. Some of these will be changed in the next versions of AirCompSim.
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--userC', type=int, required=False)
    parser.add_argument('--locN', type=int, required=False)
    args = parser.parse_args()
    #print("Arfs: ", args.userC)

    numberOfUsers = [20, 40, 60, 80, 100] #[args.userC]
    numberOfServers = [4]
    numberOfUAVs = [0, 5, 10, 15, 20]
    uavWaitingPolicy = [100]

    #edgeServerRadius = [50, 100, 150, 200]
    uavRadius = [100] #[10, 15, 20, 25, 30]
    isDRL = False
    isDRLTraining = False
    locationsDRL = []
    numberOfLocations = 3
    if args.locN:
        numberOfLocations = args.locN


    userMobilityPolicy = ["Mobile"]  # Nomadic, Mobile
    numberOfEpisodes = 500  # this can also be used as number of episodes in the simulator
    repeatCount = 1
    simulationCount = 1
    print("Simulation is starting...")
    logging.basicConfig(filename="AirSim.log", level=logging.INFO)
    # logging.basicConfig(filename="AirSimDebug.log", level=logging.DEBUG)
    logging.disable(logging.INFO)
    logging.info("Simulation log is started")
    # logging.disable(logging.DEBUG)


    drlScores = []
    agent = []  # multiple-agent solutions can be investigated in this way

    edgeResults = pd.DataFrame()
    appResults = pd.DataFrame()
    uavResults = pd.DataFrame()
    scenarioResults = pd.DataFrame()


    searcherUAVLocations = []
    single_searcherUAVLocations = []



    for seedNumber in range(0, 1):


        if isDRLTraining: # isDRL
            agent = []  # multiple-agent solutions can be investigated in this way
            #connectionThreshold = 5 # 1: worst case, take everyone!
            detectedLocations = []

            highestScore = -10000
            anAgent = DDQNAgent(state_size=2, action_size=5)
            agent.append(anAgent)
            results = pd.DataFrame()
            singleEpisodeResults = []
            for _ in range(numberOfEpisodes):
                simulation = Simulation(userCount=numberOfUsers[0],
                                        edgeCount=1,
                                        testNumber=simulationCount,
                                        uavCount=1,
                                        flyPolicy="DRL",
                                        waitingPolicy=100,
                                        userMobilityPolicy="Fixed",
                                        agent=agent,
                                        uavRadius=uavRadius[0],
                                        seedNo=seedNumber,
                                        isDRLTraining=isDRLTraining,
                                        locations=detectedLocations)

                simResultsTraining = simulation.StartSimulation()
                #print("Result of seedNo ", seedNumber, " is calculating...")
                appResultsTrainingDf = simResultsTraining[0]
                singleEpisodeResults.append(appResultsTrainingDf["DRLScore"][0])
                if "DRLScore" in appResultsTrainingDf.columns and len(appResultsTrainingDf) > 0 and appResultsTrainingDf["DRLScore"][0] > highestScore:
                    highestScore = appResultsTrainingDf["DRLScore"][0]
                    results = appResultsTrainingDf


        uavFlyPolicy = ["LSI"]

        totalSims = len(numberOfUsers) * len(numberOfServers) * len(numberOfUAVs) * len(uavFlyPolicy) * len(
            uavWaitingPolicy) * repeatCount * len(uavRadius)


        for anUavFlyingPolicy in uavFlyPolicy:
            for anUavWaitingPolicy in uavWaitingPolicy:
                for aUserMobilityPolicy in userMobilityPolicy:
                    for edgeCount in numberOfServers:
                        for uavCount in numberOfUAVs:
                            for userCount in numberOfUsers:
                                for _ in range(repeatCount):
                                    logging.info("Simulation %s has been started.", str(simulationCount))
                                    print("For userCount", userCount, " simulation ", simulationCount, " has been started...")
                                    simulation = Simulation(userCount=userCount,
                                                            edgeCount=edgeCount,
                                                            testNumber=simulationCount,
                                                            uavCount=uavCount,
                                                            flyPolicy=anUavFlyingPolicy,
                                                            waitingPolicy=anUavWaitingPolicy,
                                                            userMobilityPolicy=aUserMobilityPolicy,
                                                            agent=agent,
                                                            uavRadius=uavRadius[0],
                                                            seedNo=seedNumber,
                                                            isDRLTraining=False,
                                                            locations=locationsDRL)
                                    simResults = simulation.StartSimulation()  # it returns dataframes in an arraylist
                                    # simResults.append(simResultDf)
                                    appResultsDf = simResults[0]
                                    edgeResultsDf = simResults[1]
                                    uavResultsDf = simResults[2]
                                    scenarioResultsDf = simResults[3]

                                    edgeResults = pd.concat([edgeResults, edgeResultsDf])
                                    appResults = pd.concat([appResults, appResultsDf])
                                    uavResults = pd.concat([uavResults, uavResultsDf])
                                    scenarioResults = pd.concat([scenarioResults, scenarioResultsDf])
                                    logging.info("Simulation %s finished", str(simulationCount))
                                    print("For userCount", userCount, " simulation ", simulationCount, " is finished.")
                                    print("% ", (simulationCount / totalSims) * 100, " is completed")
                                    print(" ")
                                    print("*********************************************")
                                    print(" ")
                                    simulationCount += 1



    # if isDRL and isDRLTraining:
    #     for anAgent in agent:
    #         anAgent.saveModel()

    # plots, statistics etc.
    if args.userC:
        appResults.to_csv("AppResults_Users_"+str(args.userC)+".csv")
    else:
        appResults.to_csv("AppResults.csv")
    edgeResults.to_csv("EdgeResults.csv")
    uavResults.to_csv("UavResults.csv")
    scenarioResults.to_csv("ScenarioResults.csv")









