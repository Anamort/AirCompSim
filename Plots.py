import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
This module provides to plot the results based on the produced .csv files after the experiments. 
It can be easily extended to different calculations based on the conducted research.
'''
class Plots(object):
    def __init__(self, numberOfServers,
                 numberOfUsers,
                 numberOfUAVs,
                 uavFlyPolicy,
                 uavWaitingPolicy,
                 ):
        self.appResults = pd.read_csv("AppResults.csv")
        self.edgeResults = pd.read_csv("EdgeResults.csv")
        self.uavResults = pd.read_csv("UavResults.csv")
        self.scenarioResults = pd.read_csv("ScenarioResults.csv")

        self.numberOfEdgeServersList = numberOfServers
        self.numberOfUsersList = numberOfUsers
        self.numberOfUAVsList = numberOfUAVs
        self.uavFlyPoliciesList = uavFlyPolicy
        self.uavWaitingPoliciesList = uavWaitingPolicy
        self.apps = ["Entertainment", "Multimedia", "Rendering", "ImageClassification"]


    def getEdgeCloudUAVRatio(self, numberOfUAVs):
        edge = []
        uav = []
        cloud = []


        for i, numberOfUsers in enumerate(self.numberOfUsersList):

            testRes = self.appResults.loc[(self.appResults["NumberOfUsers"] == numberOfUsers)
                                          & (self.appResults["NumberOfUAVs"] == numberOfUAVs)
                                          & (self.appResults["UAVWaitingPolicy"] == 100)]
            edge.append(testRes["OffloadedToEdge"].sum() / testRes["TotalTasks"].sum())
            uav.append(testRes["OffloadedToUAV"].sum() / testRes["TotalTasks"].sum())
            cloud.append(testRes["OffloadedToCloud"].sum() / testRes["TotalTasks"].sum())

            #print("Normal: ", testRes["TotalTasks"].sum() / len(testRes["TestNo"].unique()))

        barWidth = 0.25
        # Set position of bar on X axis
        br1 = np.arange(len(self.numberOfUsersList))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]

        plt.figure()
        plt.bar(br1, edge, width=barWidth, label="Edge")
        plt.bar(br2, uav, width=barWidth, label="UAV")
        plt.bar(br3, cloud, width=barWidth, label="Cloud")
        plt.legend()
        plt.xlabel("Number of Users")
        plt.ylabel("Offloaded Tasks (%)")
        plt.ylim(0, 1.05)
        plt.xticks([r + barWidth for r in range(len(self.numberOfUsersList))], self.numberOfUsersList)

        plt.savefig("OffloadedTaskPercentage-"+str(numberOfUAVs)+"-UAVs.pdf")



    def getNumberOfTasks(self):
        res = []

        for i, numberOfUsers in enumerate(self.numberOfUsersList):

            testRes = self.appResults.loc[(self.appResults["NumberOfUsers"] == numberOfUsers)
                                          & (self.appResults["UAVWaitingPolicy"] == 100)]
            res.append(testRes["TotalTasks"].sum() / len(testRes["TestNo"].unique()))


        plt.figure()

        plt.plot(self.numberOfUsersList, res)

        plt.xlabel("Number of Users")
        plt.ylabel("Avg Number of Tasks")

        plt.savefig("TotalTask.pdf")


    def getAppResults(self, numberOfUsers):
        res = np.zeros((len(self.numberOfUAVsList), len(self.apps)))

        for i, numberOfUAVs in enumerate(self.numberOfUAVsList):
            for j, app in enumerate(self.apps):
                testRes = self.appResults.loc[(self.appResults["ApplicationTypes"] == app) &
                                              (self.appResults["NumberOfUsers"] == numberOfUsers)
                                               & (self.appResults["NumberOfUAVs"] == numberOfUAVs)
                                               & (self.appResults["UAVWaitingPolicy"] == 100)]
                res[i, j] = (testRes["SuccessfulTasks"].sum() / testRes["TotalTasks"].sum()) * 100

        plt.figure()

        for i in range(0, len(self.apps)):
            plt.plot(self.numberOfUAVsList, res[:, i], label=(self.apps[i]))

        plt.legend()
        plt.xlabel("Number of UAVs")
        plt.ylabel("Avg Task Success Rate")
        plt.ylim(0, 105)
        plt.xticks(self.numberOfUAVsList)

        plt.savefig("AppBasedTaskSuccess-"+str(numberOfUsers)+"-Users.pdf")


    def getGeneralResults(self, uavWaitingTime):
        res = np.zeros((len(self.numberOfUsersList), len(self.numberOfUAVsList)))

        for i, numberOfUsers in enumerate(self.numberOfUsersList):
            for j, numberOfUAVs in enumerate(self.numberOfUAVsList):
                testRes = self.appResults.loc[(self.appResults["NumberOfUsers"] == numberOfUsers)
                                              & (self.appResults["NumberOfUAVs"] == numberOfUAVs)
                                              & (self.appResults["UAVWaitingPolicy"] == uavWaitingTime)]
                res[i, j] = (testRes["SuccessfulTasks"].sum() / testRes["TotalTasks"].sum()) * 100


        plt.figure()

        for i in range(0, len(self.numberOfUAVsList)):
            plt.plot(self.numberOfUsersList, res[:, i], label=(str(self.numberOfUAVsList[i]) + " UAVs"))

        plt.legend()
        plt.xlabel("Number of Users")
        plt.ylabel("Avg Task Success Rate")
        plt.ylim(0, 105)

        plt.savefig("Overall-res-waiting-"+str(uavWaitingTime)+"-time.pdf")


    def getEdgeUtilization(self):
        res = np.zeros((len(self.numberOfUsersList), len(self.numberOfUAVsList)))

        for i, numberOfUsers in enumerate(self.numberOfUsersList):
            for j, numberOfUAVs in enumerate(self.numberOfUAVsList):
                testRes = self.edgeResults.loc[(self.edgeResults["NumberOfUsers"] == numberOfUsers)
                                              & (self.edgeResults["NumberOfUAVs"] == numberOfUAVs)
                                              & (self.edgeResults["UAVWaitingPolicy"] == 100)]
                res[i, j] = testRes["EdgeUtilization"].mean()

        plt.figure()

        for i in range(0, len(self.numberOfUAVsList)):
            plt.plot(self.numberOfUsersList, res[:, i], label=(str(self.numberOfUAVsList[i]) + " UAVs"))

        plt.legend()
        plt.xlabel("Number of Users")
        plt.ylabel("Avg Edge Utilization")
        plt.ylim(0, 105)

        plt.savefig("EdgeUtilization.pdf")

    def getUAVUtilization(self):
        res = np.zeros((len(self.numberOfUsersList), len(self.numberOfUAVsList)))

        for i, numberOfUsers in enumerate(self.numberOfUsersList):
            for j, numberOfUAVs in enumerate(self.numberOfUAVsList):
                testRes = self.uavResults.loc[(self.uavResults["NumberOfUsers"] == numberOfUsers)
                                              & (self.uavResults["NumberOfUAVs"] == numberOfUAVs)
                                              & (self.uavResults["UAVWaitingPolicy"] == 100)]
                res[i, j] = testRes["UAVUtilization"].mean()

        plt.figure()

        for i in range(0, len(self.numberOfUAVsList)):
            plt.plot(self.numberOfUsersList, res[:, i], label=(str(self.numberOfUAVsList[i]) + " UAVs"))

        plt.legend()
        plt.xlabel("Number of Users")
        plt.ylabel("Avg UAV Utilization")
        plt.ylim(0, 105)

        plt.savefig("UAVUtilization.pdf")

    def getAvgServiceTime(self):
        # QueueingDelays

        res = np.zeros((len(self.numberOfUsersList), len(self.numberOfUAVsList)))

        for i, numberOfUsers in enumerate(self.numberOfUsersList):
            for j, numberOfUAVs in enumerate(self.numberOfUAVsList):
                testRes = self.appResults.loc[(self.appResults["NumberOfUsers"] == numberOfUsers)
                                              & (self.appResults["NumberOfUAVs"] == numberOfUAVs)
                                              & (self.appResults["UAVWaitingPolicy"] == 100)]
                res[i, j] = testRes["QueueingDelays"].mean()

        plt.figure()

        for i in range(0, len(self.numberOfUAVsList)):
            plt.plot(self.numberOfUsersList, res[:, i], label=(str(self.numberOfUAVsList[i]) + " UAVs"))

        plt.legend()
        plt.xlabel("Number of Users")
        plt.ylabel("Avg Service Time (s)")

        plt.savefig("AvgServiceTime.pdf")




if __name__ == '__main__':
    # TODO: Take these from a configuration file based on a scenario
    numberOfUsers = [20, 40, 60, 80, 100]
    numberOfServers = [4]
    numberOfUAVs = [0, 5, 10, 15, 20]
    uavFlyPolicy = ["LSI"]
    uavWaitingPolicy = [100]  # seconds

    #edgeServerRadius = [50, 100, 150, 200]
    uavRadius = [80]

    userMobilityPolicy = ["Mobile"]

    plots = Plots(numberOfServers=numberOfServers,
                  numberOfUsers=numberOfUsers,
                  numberOfUAVs=numberOfUAVs,
                  uavFlyPolicy=uavFlyPolicy,
                  uavWaitingPolicy=uavWaitingPolicy)

    for waitingTime in uavWaitingPolicy:
        plots.getGeneralResults(waitingTime)

    plots.getEdgeUtilization()
    plots.getUAVUtilization()
    plots.getAvgServiceTime()
    for userCount in numberOfUsers:
        plots.getAppResults(userCount)

    plots.getNumberOfTasks()
    for uavCount in numberOfUAVs:
        plots.getEdgeCloudUAVRatio(uavCount)












