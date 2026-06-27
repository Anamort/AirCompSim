from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    number_of_users: List[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])
    number_of_servers: List[int] = field(default_factory=lambda: [4])
    number_of_uavs: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20])
    uav_waiting_policy: List[int] = field(default_factory=lambda: [100])
    uav_radius: List[int] = field(default_factory=lambda: [100])
    uav_fly_policy: List[str] = field(default_factory=lambda: ["LSI"])
    user_mobility_policy: List[str] = field(default_factory=lambda: ["Mobile"])
    repeat_count: int = 50
    simulation_time: float = 1000.0
    seed: Optional[int] = None
    is_drl: bool = False
    is_drl_training: bool = False
    output_dir: Optional[str] = None
    generate_edge_radius_plot: bool = False
    write_root_csvs: bool = False

    @classmethod
    def smoke(cls) -> ExperimentConfig:
        return cls(
            number_of_users=[20],
            number_of_uavs=[0, 5],
            repeat_count=1,
            seed=42,
            generate_edge_radius_plot=True,
        )

    @classmethod
    def paper(cls) -> ExperimentConfig:
        return cls(
            number_of_users=[20, 40, 60, 80, 100],
            number_of_servers=[4],
            number_of_uavs=[0, 5, 10, 15, 20],
            uav_waiting_policy=[100],
            uav_radius=[100],
            uav_fly_policy=["LSI"],
            user_mobility_policy=["Mobile"],
            repeat_count=50,
            simulation_time=1000.0,
            generate_edge_radius_plot=True,
        )

    def total_simulations(self) -> int:
        return (
            len(self.number_of_users)
            * len(self.number_of_servers)
            * len(self.number_of_uavs)
            * len(self.uav_fly_policy)
            * len(self.uav_waiting_policy)
            * len(self.uav_radius)
            * len(self.user_mobility_policy)
            * self.repeat_count
        )

    def ensure_output_dir(self) -> Path:
        if self.output_dir:
            path = Path(self.output_dir)
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path("results") / run_id
            self.output_dir = str(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
