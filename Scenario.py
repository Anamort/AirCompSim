
from User import User
from Application import Application, Task, OffloadEntity, ApplicationType
from Location import Location
from Server import Server, EdgeServer, UAV, CloudServer
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from Event import Event, EventType

'''
This module provides the corresponding scenarios including the existing servers and their capacities, application
profiles, number of users, etc. Moreover, you can define a dynamic scenario using updateScenario method if you want to
observe the results of a dynamic event such as a server failure. 
'''


class Scenario(object):
    def __init__(self,
                 numberOfUAVs,
                 numberOfUsers,
                 testNo,
                 UAVWaitingPolicy,
                 UAVFlyPolicy,
                 uavRadius,
                 scenario="BasicEdge",
                 edgeCapacity=1000,
                 uavCapacity= 500,
                 edgeRadius=100,
                 edgePower=100,
                 uavPower=100,
                 altitude=200):
        self.numberOfUsers = numberOfUsers
        self.testNumber = testNo
        self.scenarioName = scenario
        self.edgeCapacity = edgeCapacity
        self.uavCapacity = uavCapacity
        self.edgeRadius = edgeRadius
        self.uavRadius = uavRadius
        self.edgePower = edgePower
        self.uavPower = uavPower
        self.altitude = altitude
        self.numberOfUAVs = numberOfUAVs
        self.UAVWaitingPolicy = UAVWaitingPolicy
        self.UAVFlyPolicy = UAVFlyPolicy
        self.userMobility = True
        self.uavMobilityTest = False
        #self.locations = []  # this is used for edge loca


        self.DRLForUAVs = False
        self.isEartquakeHappened = False
        self.isDoubled = False  # for earthquake scenario
        self.isDRL = False  # for DRL-based studies
        self.isMovementStart = False


    def basicEdgeScenario(self):

        edgeLocation1 = Location(x=100, y=300, z=0)
        edgeLocation2 = Location(x=300, y=300, z=0)
        edgeLocation3 = Location(x=100, y=100, z=0)
        edgeLocation4 = Location(x=300, y=100, z=0)


        edgeLocationsX = [100, 300, 100, 300]
        edgeLocationsY = [300, 300, 100, 100]
        plt.figure()
        plt.scatter(edgeLocationsX, edgeLocationsY, color="red")  # , alpha=0.2, markersize=50
        for uavX, uavY in zip(edgeLocationsX, edgeLocationsY):
            circle1 = plt.Circle((uavX, uavY), self.edgeRadius, color='red', alpha=0.2)
            plt.gca().add_patch(circle1)
        plt.xlim(0, 400)
        plt.ylim(0, 400)

        plt.savefig("Edge-Radius.pdf")


        EdgeServer(capacity=self.edgeCapacity, location=edgeLocation1, radius=self.edgeRadius, power=self.edgePower)
        EdgeServer(capacity=self.edgeCapacity, location=edgeLocation2, radius=self.edgeRadius, power=self.edgePower)
        EdgeServer(capacity=self.edgeCapacity, location=edgeLocation3, radius=self.edgeRadius, power=self.edgePower)
        EdgeServer(capacity=self.edgeCapacity, location=edgeLocation4, radius=self.edgeRadius, power=self.edgePower)


        entertainmentApp = ApplicationType(name="Entertainment", cpuCycle=100, worstDelay=0.3, bestDelay=0.1,
                                           exponential=10)

        multimediaApp = ApplicationType(name="Multimedia", cpuCycle=100, worstDelay=3, bestDelay=0.1,
                                        exponential=10)

        imageRenderingApp = ApplicationType(name="Rendering", cpuCycle=200, worstDelay=1, bestDelay=0.5,
                                        exponential=20)

        augmentedReality = ApplicationType(name="ImageClassification", cpuCycle=600, worstDelay=1, bestDelay=1,
                                        exponential=20)





        for _ in range(self.numberOfUsers):
            aUser = User(location=Location(x=np.random.uniform(low=1, high=399),
                                           y=np.random.uniform(low=1, high=399),
                                           z=0))
            aUser.setApplication(Application(entertainmentApp, simTime=0))
            aUser.setApplication(Application(multimediaApp, simTime=0))
            aUser.setApplication(Application(imageRenderingApp, simTime=0))
            aUser.setApplication(Application(augmentedReality, simTime=0))


        for _ in range(self.numberOfUAVs):
            UAV(location=Location(x=0,
                                  y=0,
                                  z=self.altitude),
                capacity=self.uavCapacity,
                radius=self.uavRadius,
                power=self.uavPower)



    def updateScenario(self, simTime, heapq, eventQueue):
        if self.scenarioName == "ADynamicScenario":
            if simTime >= 1000:
                logging.info("A dynamic event is happened!")


        elif self.scenarioName == "AnotherScenario":
            if simTime >= 2000:
                logging.info("Another dynamic event is happened!")

    def DRLScenario(self, seedNumber):
        pass



    def computeMetrics(self):
        if self.scenarioName == "ADynamicScenario" or self.scenarioName == "AnotherScenario":
            logging.info("Scenario specific metrics are computed.")

        elif self.scenarioName == "TestMobility":
            colors = ["blue", "red", "green"]
            counter = 0
            for user in User.users:
                trajectory = user.trajectory
                xVals = []
                yVals = []
                for loc in trajectory:
                    xVals.append(loc.x)
                    yVals.append(loc.y)

                plt.scatter(xVals, yVals, color=colors[counter])
                counter += 1
            plt.savefig("user-paths.pdf")




    def uavTestScenario(self):
        for _ in range(3):
            UAV(location=Location(x=np.random.uniform(low=1, high=999),
                                  y=np.random.uniform(low=1, high=999), z=self.altitude), capacity=self.uavCapacity,
                radius=self.uavRadius, power=self.uavPower)



    def mobilityTestScenario(self):
        aUser = User(
            location=Location(x=np.random.uniform(low=0, high=999), y=np.random.uniform(low=0, high=999), z=0))

        aUser = User(
            location=Location(x=np.random.uniform(low=0, high=999), y=np.random.uniform(low=0, high=999), z=0))

        aUser = User(
            location=Location(x=np.random.uniform(low=0, high=999), y=np.random.uniform(low=0, high=999), z=0))



