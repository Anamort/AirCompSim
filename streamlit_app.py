from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from experiment_config import ExperimentConfig
from experiment_runner import run_experiment
from Plots import generate_plots as build_plots


PRESETS = {
    "Smoke test": ExperimentConfig.smoke(),
    "Paper scenario": ExperimentConfig.paper(),
    "Custom": ExperimentConfig(),
}


def parse_int_list(value: str, default: list[int]) -> list[int]:
    cleaned = value.replace("[", "").replace("]", "")
    if not cleaned.strip():
        return default
    return [int(item.strip()) for item in cleaned.split(",") if item.strip()]


def parse_str_list(value: str, default: list[str]) -> list[str]:
    cleaned = value.replace("[", "").replace("]", "")
    if not cleaned.strip():
        return default
    return [item.strip().strip('"').strip("'") for item in cleaned.split(",") if item.strip()]


def summarize_results(app_results: pd.DataFrame) -> dict[str, float]:
    if app_results.empty:
        return {
            "task_success_rate": 0.0,
            "avg_service_time": 0.0,
            "total_tasks": 0.0,
        }

    total_tasks = app_results["TotalTasks"].sum()
    successful_tasks = app_results["SuccessfulTasks"].sum()
    success_rate = (successful_tasks / total_tasks * 100) if total_tasks else 0.0
    avg_service_time = app_results["QueueingDelays"].mean() if len(app_results) else 0.0
    return {
        "task_success_rate": success_rate,
        "avg_service_time": avg_service_time,
        "total_tasks": float(total_tasks),
    }


def list_pdf_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*.pdf"))


st.set_page_config(page_title="AirCompSim", layout="wide")
st.title("AirCompSim Experiment Dashboard")
st.caption("Configure and run air computing simulations without editing main.py")

with st.sidebar:
    st.header("Experiment preset")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    config = PRESETS[preset_name]

    st.header("Sweep parameters")
    users = st.text_input("Number of users", value=",".join(str(v) for v in config.number_of_users))
    uavs = st.text_input("Number of UAVs", value=",".join(str(v) for v in config.number_of_uavs))
    servers = st.text_input("Number of edge servers", value=",".join(str(v) for v in config.number_of_servers))
    waiting_policies = st.text_input("UAV waiting policy (seconds)", value=",".join(str(v) for v in config.uav_waiting_policy))
    uav_radius = st.text_input("UAV radius", value=",".join(str(v) for v in config.uav_radius))
    fly_policies = st.text_input("UAV fly policies", value=",".join(config.uav_fly_policy))
    mobility_policies = st.text_input("User mobility policies", value=",".join(config.user_mobility_policy))

    st.header("Run settings")
    repeat_count = st.number_input("Repeat count", min_value=1, max_value=500, value=config.repeat_count)
    simulation_time = st.number_input("Simulation time (seconds)", min_value=100, max_value=10000, value=int(config.simulation_time))
    seed_value = st.text_input("Random seed (optional)", value="" if config.seed is None else str(config.seed))
    output_dir = st.text_input("Output directory (optional)", value=config.output_dir or "")
    generate_edge_plot = st.checkbox("Generate edge radius plot", value=config.generate_edge_radius_plot)
    should_generate_plots = st.checkbox("Generate PDF plots after run", value=True)

run_col, info_col = st.columns([1, 2])
with info_col:
    st.info(
        "Smoke test is recommended first. The full paper scenario with 50 repeats runs "
        "1,250 simulations and may take a long time."
    )

if run_col.button("Run experiment", type="primary"):
    try:
        run_config = ExperimentConfig(
            number_of_users=parse_int_list(users, config.number_of_users),
            number_of_servers=parse_int_list(servers, config.number_of_servers),
            number_of_uavs=parse_int_list(uavs, config.number_of_uavs),
            uav_waiting_policy=parse_int_list(waiting_policies, config.uav_waiting_policy),
            uav_radius=parse_int_list(uav_radius, config.uav_radius),
            uav_fly_policy=parse_str_list(fly_policies, config.uav_fly_policy),
            user_mobility_policy=parse_str_list(mobility_policies, config.user_mobility_policy),
            repeat_count=int(repeat_count),
            simulation_time=float(simulation_time),
            seed=int(seed_value) if seed_value.strip() else None,
            output_dir=output_dir.strip() or None,
            generate_edge_radius_plot=generate_edge_plot,
        )

        total_runs = run_config.total_simulations()
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(current: int, total: int, message: str) -> None:
            progress.progress(current / total)
            status.write(f"Run {current}/{total}: {message}")

        with st.spinner(f"Running {total_runs} simulations..."):
            app_results, edge_results, uav_results, scenario_results, results_path = run_experiment(
                run_config,
                progress_callback=on_progress,
            )
            if should_generate_plots:
                build_plots(run_config, results_path)

        st.session_state["last_results_dir"] = str(results_path)
        st.session_state["run_config"] = {
            "number_of_users": run_config.number_of_users,
            "number_of_uavs": run_config.number_of_uavs,
            "number_of_servers": run_config.number_of_servers,
            "repeat_count": run_config.repeat_count,
            "simulation_time": run_config.simulation_time,
            "seed": run_config.seed,
        }
        st.session_state["app_results"] = app_results
        st.session_state["edge_results"] = edge_results
        st.session_state["uav_results"] = uav_results
        st.session_state["scenario_results"] = scenario_results
        st.success(f"Experiment completed. Results saved to `{results_path}`")
    except Exception as exc:
        st.error(f"Experiment failed: {exc}")

if "app_results" in st.session_state:
    st.subheader("Summary metrics")
    summary = summarize_results(st.session_state["app_results"])
    metric_cols = st.columns(3)
    metric_cols[0].metric("Task success rate (%)", f"{summary['task_success_rate']:.2f}")
    metric_cols[1].metric("Avg service time (s)", f"{summary['avg_service_time']:.4f}")
    metric_cols[2].metric("Total tasks", f"{summary['total_tasks']:.0f}")

    st.subheader("Application results")
    st.dataframe(st.session_state["app_results"], use_container_width=True)

    st.subheader("Edge results")
    st.dataframe(st.session_state["edge_results"], use_container_width=True)

    st.subheader("UAV results")
    st.dataframe(st.session_state["uav_results"], use_container_width=True)

    results_dir = Path(st.session_state.get("last_results_dir", "results"))
    if results_dir.exists():
        st.subheader("Generated files")
        for csv_file in sorted(results_dir.glob("*.csv")):
            st.download_button(
                label=f"Download {csv_file.name}",
                data=csv_file.read_bytes(),
                file_name=csv_file.name,
                mime="text/csv",
            )
        for pdf_file in list_pdf_files(results_dir):
            st.download_button(
                label=f"Download {pdf_file.name}",
                data=pdf_file.read_bytes(),
                file_name=pdf_file.name,
                mime="application/pdf",
            )

    with st.expander("Saved run configuration"):
        st.code(json.dumps(st.session_state.get("run_config", {}), indent=2), language="json")
