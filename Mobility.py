import random

from Server import UAV, EdgeServer
from main import SimulationBoundry
from random import randrange
from User import User
from Location import Location
from Application import Application, Task
import numpy as np
import logging
import math

'''
This module provides the user mobility and also UAV policies. It will be redesigned to seperate those. Some of the
following methods are experimental.
'''
class Mobility(object):
    def __init__(self, userMobilityModel, uavFlyingMode, simBoundry: SimulationBoundry):
        self.userMobilityModel = userMobilityModel
        self.uavFlyingMode = uavFlyingMode
        self.simBoundry = simBoundry
        #self.clusteringInterval = 300

    '''
    This method returns a random location for given user to move based on Random Waypoint Model.
    '''
    @classmethod
    def moveUser(cls, currentLoc: Location, simBoundry: SimulationBoundry, radius):

        # get random number between 0 and radius for x and y

        newLoc = Location(x=np.random.uniform(low=currentLoc.x - radius, high=currentLoc.x + radius),
                          y=np.random.uniform(low=currentLoc.y - radius, high=currentLoc.y + radius),
                          z=0)

        while not simBoundry.isInBoundry(newLoc.x, newLoc.y, newLoc.z):
            newLoc = Location(x=np.random.uniform(low=currentLoc.x - radius, high=currentLoc.x + radius),
                              y=np.random.uniform(low=currentLoc.y - radius, high=currentLoc.y + radius),
                              z=0)

        return newLoc

    '''
        This method returns next locations of UAVs considering their closest location. Thus, sorting is performed in here.
    '''
    @classmethod
    def sortUAVsForClusterLocations(cls, clusterLocations):
        if not clusterLocations:
            return []
        uavLocationsToGo = []
        uavs = UAV.uavs
        selectedClusters = set()
        for i in range(len(clusterLocations)):
            uav = uavs[i]
            minDistance = 999999
            minLocation: Location = None
            selectedClusterNo = 0
            for clusterNo in range(len(clusterLocations)):
                if clusterNo not in selectedClusters:
                    distance = Location.getEuclideanDistance2D(uav.location, clusterLocations[clusterNo])
                    if distance < minDistance:
                        minDistance = distance
                        minLocation = clusterLocations[clusterNo]
                        selectedClusterNo = clusterNo

            selectedClusters.add(selectedClusterNo)
            uavLocationsToGo.append(clusterLocations[selectedClusterNo])
            logging.info("Final uav location to go: %s ", str(clusterLocations[selectedClusterNo]))
        return uavLocationsToGo



    @classmethod
    def clusterBasedUAV(cls, simBoundry: SimulationBoundry):
        # run kmeans
        userClusters = {}
        logging.info("clusterBasedUAV (User-based): Clustering has been started...")
        numberOfUAVs = len(UAV.uavs)  # this will be the number of clusters
        clusterLocations = []  # central points
        logging.info("taskBasedClusteringForUAV (User-based): Initial cluster locations are:")
        for clusterNo in range(numberOfUAVs):
            loc = Location(x=np.random.uniform(low=100, high=600),
                           y=np.random.uniform(low=50, high=350),
                           z=0)
            logging.info("clusterBasedUAV (User-based): Initial location for clusterNo %s is %s", str(clusterNo), loc)
            clusterLocations.append(loc)

        logging.info("clusterBasedUAV (User-based): Clustering for 20 loop is starting...")
        for i in range(10):

            # resetting user clusters
            for clusterNo in range(numberOfUAVs):
                userClusters[clusterNo] = []

            for user in User.users:
                userLoc = user.getLocation()
                minManDistance = Location.getEuclideanDistance2D(userLoc, clusterLocations[0])
                userClusterNo = 0
                logging.info("clusterBasedUAV (User-based): For user id %s with location at %s, initial min. Manhattan Distance is %s based on cluster location[0] at %s",
                             str(user.id), user.getLocation(), str(minManDistance), clusterLocations[0])
                for clusterNo in range(len(clusterLocations)):
                    clusterLoc = clusterLocations[clusterNo]
                    manDistance = Location.getEuclideanDistance2D(userLoc, clusterLoc)
                    logging.info("clusterBasedUAV (User-based): Manhattan Distance of cluster %s at location %s to user %s at location %s is %s",
                                 str(clusterNo), clusterLocations[clusterNo], str(user.id), user.getLocation(), str(manDistance))
                    if manDistance < minManDistance:
                        minManDistance = manDistance
                        userClusterNo = clusterNo

                userClusters[userClusterNo].append(userLoc)
                logging.info("clusterBasedUAV (User-based): For user %s, the selected user cluster is %s . The user location is at %s and the cluster location is at %s",
                             str(user.id), str(userClusterNo), user.getLocation(), clusterLocations[userClusterNo])


            for clusterNo in range(len(clusterLocations)):
                usersLoc = userClusters[clusterNo]
                avgX:float = 0
                avgY:float = 0
                for loc in usersLoc:
                    avgX += loc.x
                    avgY += loc.y
                if len(usersLoc) > 0:
                    avgX = avgX / len(usersLoc)
                    avgY = avgY / len(usersLoc)
                else:
                    # to prevent empty cluster problem
                    avgX = np.random.uniform(low=100, high=600)
                    avgY = np.random.uniform(low=50, high=350)
                clusterLocations[clusterNo] = Location(x=avgX, y=avgY, z=0)
                logging.info("clusterBasedUAV (User-based): Selected location for clusterNo %s is %s at iteration %s", str(clusterNo), clusterLocations[clusterNo], str(i))

        return clusterLocations  # UAVs should go these locations


    @classmethod
    # This is an experimental method
    def newTaskBasedClustering(cls, numberOfClusters):

        clusterNoToUsers = {}  # cluster no ----> [user]

        # TODO: This should be parameterized
        # clusterLocations = [Location(x=150, y=100, z=0), Location(x=600, y=300, z=0),
        #                     Location(x=400, y=800, z=0)]  # central points

        clusterLocations = [Location(x=80, y=30, z=0), Location(x=30, y=80, z=0)]  # central points for dqn

        for i in range(numberOfClusters):  # considering citys
            clusterNoToUsers[i] = []

        for user in User.users:
            if user.city == "FirstCity":
                clusterNoToUsers[0].append(user)
            elif user.city == "SecondCity":
                clusterNoToUsers[1].append(user)
            else:
                clusterNoToUsers[2].append(user)

        finalClusterLocations = []
        numberOfTasksOfClusters = [0 for _ in range(len(clusterLocations))]
        for clusterNo in range(len(clusterLocations)):
            users = clusterNoToUsers[clusterNo]
            for user in users:
                for app in user.applications:
                    numberOfTasksOfClusters[clusterNo] += len(app.tasks)


        # Should perform load balancing
        for _ in range(len(UAV.uavs)):
            crowdedClusterNo = 0
            highestNumberOfTasks = numberOfTasksOfClusters[crowdedClusterNo]

            for clusterNo in range(numberOfClusters):
                if numberOfTasksOfClusters[clusterNo] > highestNumberOfTasks:
                    crowdedClusterNo = clusterNo
                    highestNumberOfTasks = numberOfTasksOfClusters[clusterNo]

            finalClusterLocations.append(clusterLocations[crowdedClusterNo])
            logging.info("Final cluster location for crowdedClusterNo %s :  %s", str(crowdedClusterNo), str(clusterLocations[crowdedClusterNo]))
            numberOfTasksOfClusters[crowdedClusterNo] -= 200
            logging.info("200 is subtracted from crowdedClusterNo %s ", str(crowdedClusterNo))

        return finalClusterLocations


    @classmethod
    def computeLSI(cls, clusterLocation, users):
        numberOfUsers = len(users)
        numberOfEdgeServers = 0
        avgTaskRate = 0
        avgEdgeCapacity = 0
        avgUAVCapacity = 0
        numberOfUAVs = 0
        requiredCapacity = 0
        totalLoad = 0

        for edgeServer in EdgeServer.edgeServers:
            if edgeServer.isInCoverage(clusterLocation):
                numberOfEdgeServers += 1
                avgEdgeCapacity += edgeServer.capacity

        #for uav in UAV.uavs:
        #    if uav.isInCoverage(clusterLocation):
        #        numberOfUAVs += 1
        #        avgUAVCapacity += uav.capacity

        totalCapacity = avgEdgeCapacity + avgUAVCapacity

        for user in users:
            for app in user.applications:
                requiredCapacity += ((1 / app.interarrivalTime) * app.cpuCycle)
                totalLoad += app.cpuCycle

        if requiredCapacity >= totalCapacity:
            return [-1, totalLoad, totalCapacity, requiredCapacity]

        lsi = totalLoad / (totalCapacity - requiredCapacity)
        return [lsi, totalLoad, totalCapacity, requiredCapacity]


    @classmethod
    def locationSelectionIndex(cls, numberOfLocations, locations, uavRadius):

        if numberOfLocations == 0:
            return []


        clusterNoToUsers = {}  # cluster no ----> [user1, user2, ...]

        clusterLocations = []
        for location in locations:
            clusterLocations.append(Location(x=location[0], y=location[1], z=0))


        for i in range(numberOfLocations):
            clusterNoToUsers[i] = []

        for user in User.users:
            index = -1
            minDistance = 9999
            for i, loc in enumerate(clusterLocations):
                dist = math.sqrt(pow(loc.x - user.currentLocation.x, 2) + pow(loc.y - user.currentLocation.y, 2))
                if dist <= minDistance and dist <= uavRadius:
                    minDistance = dist
                    index = i
            if index != -1:
                clusterNoToUsers[index].append(user)


        finalClusterLocations = []
        clustersLSI = [0 for _ in range(len(clusterLocations))]
        requiredUAVCount = [0 for _ in range(len(clusterLocations))]

        # initial evaluation
        for clusterNo in range(len(clusterLocations)):
            users = clusterNoToUsers[clusterNo]
            if len(users) == 0:
                #print("A rare case is happened (no users in defined location): location --> ", clusterLocations[clusterNo])
                #requiredUAVCount[clusterNo] = 0
                continue
            #  returns [lsi, totalLoad, totalCapacity, requiredCapacity]
            parameters = Mobility.computeLSI(clusterLocation=clusterLocations[clusterNo], users=users)
            clustersLSI[clusterNo] = parameters[0]  # lsi TODO: Make them as dictionary
            requiredDelay = Mobility.getRequiredDelayForLocation(users=users)
            if clustersLSI[clusterNo] < 0 or clustersLSI[clusterNo] > requiredDelay:
                requiredUAVCount[clusterNo] = Mobility.computeRequiredUAVCount(uavCapacity=UAV.uavs[0].capacity,
                                                                    totalLoad=parameters[1],
                                                                    totalCapacity=parameters[2],
                                                                    requiredCapacity=parameters[3],
                                                                    requiredDelay=requiredDelay) # TODO: Make this based on the scenario


        totalUAVCount = len(UAV.uavs)
        logging.info("Total uav count: %s", str(totalUAVCount))
        logging.info("Initial required UAV counts: ")
        for clusterNo in range(len(clusterLocations)):
            logging.info("ClusterNo %s needs %s UAV considering its LSI as %s", str(clusterNo), str(requiredUAVCount[clusterNo]), str(clustersLSI[clusterNo]))
        isLoadBalancing = False
        if isLoadBalancing:  # UAVs are sent to each location (cluster) regarding round-robin
            while totalUAVCount > 0 and max(requiredUAVCount) > 0:
                initialCount = totalUAVCount
                for clusterNo in range(numberOfLocations):
                    if requiredUAVCount[clusterNo] > 0:
                        finalClusterLocations.append(clusterLocations[clusterNo])
                        requiredUAVCount[clusterNo] -= 1
                        totalUAVCount -= 1
                        if totalUAVCount < 1:
                            break
                if initialCount == totalUAVCount:
                    break
        else:  # UAVs are sent to each location (cluster) based on higher needs
            while totalUAVCount > 0 and max(requiredUAVCount) > 0:
                initialCount = totalUAVCount

                clusterNo = requiredUAVCount.index(max(requiredUAVCount))
                finalClusterLocations.append(clusterLocations[clusterNo])
                requiredUAVCount[clusterNo] -= 1
                totalUAVCount -= 1

                if initialCount == totalUAVCount:
                    break


        logging.info("locationSelectionIndex: Final cluster locations for UAVs: ")
        counter = 0
        for loc in finalClusterLocations:
            logging.info("UAV with id %s will be at %s", str(counter), loc)
            counter += 1



        return finalClusterLocations

    @classmethod
    def computeRequiredUAVCount(cls, uavCapacity, totalLoad, totalCapacity, requiredCapacity, requiredDelay):
        additionalCapacityForUAV = (totalLoad/requiredDelay) + requiredCapacity - totalCapacity
        return additionalCapacityForUAV/uavCapacity

    @classmethod
    def getRequiredDelayForLocation(cls, users):
        avgWorstDelay = 0
        totalApp = 0
        for user in users:
            for app in user.applications:
                avgWorstDelay += app.worstDelay
                totalApp += 1
        return avgWorstDelay/totalApp  # TODO: Search better methods if it is possible!



    @classmethod
    def randomUAV(cls, simBoundry: SimulationBoundry):
        x = np.random.uniform(low=100, high=600)
        y = np.random.uniform(low=50, high=350)

        return Location(x, y, 0)

    @classmethod
    def randomUAVMove(cls, locations):
        loc = random.choice(locations)
        return loc
