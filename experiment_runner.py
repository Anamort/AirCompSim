from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from DDQN import DDQNAgent
from experiment_config import ExperimentConfig
from main import Simulation


ProgressCallback = Optional[Callable[[int, int, str], None]]


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_experiment(
    config: ExperimentConfig,
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    output_dir = config.ensure_output_dir()
    set_random_seed(config.seed)

    logging.basicConfig(filename=str(output_dir / "AirSim.log"), level=logging.INFO)
    logging.disable(logging.INFO)
    logging.info("Simulation log is started")

    agent = []
    edge_results = pd.DataFrame()
    app_results = pd.DataFrame()
    uav_results = pd.DataFrame()
    scenario_results = pd.DataFrame()
    locations_drl = []
    simulation_count = 1
    total_sims = config.total_simulations()

    if config.is_drl_training:
        agent = [DDQNAgent(state_size=2, action_size=5)]
        for episode in range(config.repeat_count):
            set_random_seed((config.seed or 0) + episode)
            simulation = Simulation(
                userCount=config.number_of_users[0],
                edgeCount=1,
                testNumber=simulation_count,
                uavCount=1,
                flyPolicy="DRL",
                waitingPolicy=config.uav_waiting_policy[0],
                userMobilityPolicy="Fixed",
                agent=agent,
                uavRadius=config.uav_radius[0],
                seedNo=config.seed or 0,
                isDRLTraining=True,
                locations=[],
                timeLimit=config.simulation_time,
                generateEdgeRadiusPlot=config.generate_edge_radius_plot,
                outputDir=str(output_dir),
            )
            simulation.StartSimulation()

    for fly_policy in config.uav_fly_policy:
        for waiting_policy in config.uav_waiting_policy:
            for mobility_policy in config.user_mobility_policy:
                for edge_count in config.number_of_servers:
                    for uav_count in config.number_of_uavs:
                        for user_count in config.number_of_users:
                            for repeat_index in range(config.repeat_count):
                                if progress_callback:
                                    progress_callback(
                                        simulation_count,
                                        total_sims,
                                        (
                                            f"userCount={user_count}, uavCount={uav_count}, "
                                            f"repeat={repeat_index + 1}/{config.repeat_count}"
                                        ),
                                    )

                                run_seed = None
                                if config.seed is not None:
                                    run_seed = config.seed + simulation_count
                                set_random_seed(run_seed)

                                simulation = Simulation(
                                    userCount=user_count,
                                    edgeCount=edge_count,
                                    testNumber=simulation_count,
                                    uavCount=uav_count,
                                    flyPolicy=fly_policy,
                                    waitingPolicy=waiting_policy,
                                    userMobilityPolicy=mobility_policy,
                                    agent=agent,
                                    uavRadius=config.uav_radius[0],
                                    seedNo=config.seed or 0,
                                    isDRLTraining=False,
                                    locations=locations_drl,
                                    timeLimit=config.simulation_time,
                                    generateEdgeRadiusPlot=config.generate_edge_radius_plot,
                                    outputDir=str(output_dir),
                                )
                                sim_results = simulation.StartSimulation()
                                app_results = pd.concat([app_results, sim_results[0]])
                                edge_results = pd.concat([edge_results, sim_results[1]])
                                uav_results = pd.concat([uav_results, sim_results[2]])
                                if sim_results[3] is not None:
                                    scenario_results = pd.concat([scenario_results, sim_results[3]])
                                simulation_count += 1

    app_results.to_csv(output_dir / "AppResults.csv", index=False)
    edge_results.to_csv(output_dir / "EdgeResults.csv", index=False)
    uav_results.to_csv(output_dir / "UavResults.csv", index=False)
    if not scenario_results.empty:
        scenario_results.to_csv(output_dir / "ScenarioResults.csv", index=False)
    else:
        (output_dir / "ScenarioResults.csv").write_text("")

    if config.write_root_csvs:
        app_results.to_csv("AppResults.csv", index=False)
        edge_results.to_csv("EdgeResults.csv", index=False)
        uav_results.to_csv("UavResults.csv", index=False)
        scenario_results.to_csv("ScenarioResults.csv", index=False)

    return app_results, edge_results, uav_results, scenario_results, output_dir
