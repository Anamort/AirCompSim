import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from experiment_config import ExperimentConfig


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
                 results_dir: Union[str, Path] = ".",
                 ):
        results_dir = Path(results_dir)
        self.appResults = pd.read_csv(results_dir / "AppResults.csv")
        self.edgeResults = pd.read_csv(results_dir / "EdgeResults.csv")
        self.uavResults = pd.read_csv(results_dir / "UavResults.csv")
        scenario_path = results_dir / "ScenarioResults.csv"
        self.scenarioResults = pd.DataFrame()
        if scenario_path.exists() and scenario_path.stat().st_size > 0:
            try:
                self.scenarioResults = pd.read_csv(scenario_path)
            except pd.errors.EmptyDataError:
                self.scenarioResults = pd.DataFrame()
        self.results_dir = results_dir

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

        plt.savefig(self.results_dir / ("OffloadedTaskPercentage-"+str(numberOfUAVs)+"-UAVs.pdf"))



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

        plt.savefig(self.results_dir / "TotalTask.pdf")


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

        plt.savefig(self.results_dir / ("AppBasedTaskSuccess-"+str(numberOfUsers)+"-Users.pdf"))


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

        plt.savefig(self.results_dir / ("Overall-res-waiting-"+str(uavWaitingTime)+"-time.pdf"))


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

        plt.savefig(self.results_dir / "EdgeUtilization.pdf")

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

        plt.savefig(self.results_dir / "UAVUtilization.pdf")

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

        plt.savefig(self.results_dir / "AvgServiceTime.pdf")






def generate_plots(config: ExperimentConfig, results_dir: Union[str, Path]) -> Plots:
    plots = Plots(
        numberOfServers=config.number_of_servers,
        numberOfUsers=config.number_of_users,
        numberOfUAVs=config.number_of_uavs,
        uavFlyPolicy=config.uav_fly_policy,
        uavWaitingPolicy=config.uav_waiting_policy,
        results_dir=results_dir,
    )

    for waiting_time in config.uav_waiting_policy:
        plots.getGeneralResults(waiting_time)

    plots.getEdgeUtilization()
    plots.getUAVUtilization()
    plots.getAvgServiceTime()
    for user_count in config.number_of_users:
        plots.getAppResults(user_count)

    plots.getNumberOfTasks()
    for uav_count in config.number_of_uavs:
        plots.getEdgeCloudUAVRatio(uav_count)

    return plots


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate AirCompSim plots from CSV results")
    parser.add_argument('--results-dir', type=str, default='.')
    parser.add_argument('--preset', choices=['smoke', 'paper', 'custom'], default='paper')
    args = parser.parse_args()

    if args.preset == 'smoke':
        config = ExperimentConfig.smoke()
    elif args.preset == 'paper':
        config = ExperimentConfig.paper()
    else:
        config = ExperimentConfig()

    generate_plots(config, args.results_dir)












