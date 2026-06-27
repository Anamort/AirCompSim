# GitHub Workflow for AirCompSim

This project is already connected to the public repository:

- Remote: `https://github.com/Anamort/AirCompSim.git`
- Default branch: `main`

## 1. Install and authenticate with GitHub CLI

If `gh` is not installed yet:

```bash
brew install gh
```

Then authenticate once on your machine:

```bash
gh auth login
```

Recommended choices:

- GitHub.com
- HTTPS
- Login with a web browser

Verify authentication:

```bash
gh auth status
git remote -v
```

## 2. Recommended development flow

Create a feature branch for each change:

```bash
git checkout -b feature/streamlit-ui
```

Run and validate locally:

```bash
source .venv/bin/activate
python main.py --preset smoke --plot
streamlit run streamlit_app.py
```

Commit only source and documentation changes:

```bash
git add main.py experiment_config.py experiment_runner.py streamlit_app.py
git add simulation_boundary.py Plots.py Scenario.py DRL.py DQN.py DDQN.py Mobility.py
git add requirements-runtime.txt .gitignore GITHUB_SETUP.md
git commit -m "Add configurable experiment runner and Streamlit UI"
```

Push the branch:

```bash
git push -u origin feature/streamlit-ui
```

Open a pull request:

```bash
gh pr create --title "Add experiment runner and Streamlit UI" --body "Adds configurable runs, result directories, and a local dashboard."
```

## 3. What not to commit

Generated artifacts are ignored via `.gitignore`:

- `results/`
- `*.csv`
- `*.pdf` outside `images/`
- `AirSim.log`
- `.venv/`, `__pycache__/`, `.idea/`

## 4. Full paper reproduction command

The published scenario uses 50 repeats and can take a long time:

```bash
source .venv/bin/activate
python main.py --preset paper --plot --output-dir results/paper_run
```

For faster validation:

```bash
python main.py --preset paper --repeat-count 2 --plot --output-dir results/paper_validation
```
