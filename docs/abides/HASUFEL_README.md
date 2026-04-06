> **Snapshotted from `abides-hasufel` v2.5.8.** If you've upgraded hasufel, verify this doc is still current.
>
> **Audience:** Developers and AI coding agents integrating with the ABIDES simulator.

<div id="top"></div>

# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

[![CI](https://github.com/GabrieleDiCorato/abides-hasufel/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrieleDiCorato/abides-hasufel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GabrieleDiCorato/abides-hasufel/graph/badge.svg)](https://codecov.io/gh/GabrieleDiCorato/abides-hasufel)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- TABLE OF CONTENTS -->
<ol>
  <li>
    <a href="#about-the-project">About The Project</a>
  </li>
  <li><a href="#citing-abides">Citing ABIDES</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li>
    <a href="#usage-regular">Usage (regular)</a>
    <ul>
      <li><a href="#using-a-python-script">Using a Python Script</a></li>
      <li><a href="#using-the-abides-command">Using the `abides` Command</a></li>
    </ul>
  </li>
  <li><a href="#usage-gym">Usage (Gym)</a></li>
  <li><a href="#default-available-markets-configurations">Default Available Markets Configurations</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project

ABIDES (Agent Based Interactive Discrete Event Simulator) is a general purpose multi-agent discrete event simulator. Agents exclusively communicate through an advanced messaging system that supports latency models.

The project is currently broken down into 3 parts: ABIDES-Core, ABIDES-Markets and ABIDES-Gym.

* ABIDES-Core: Core general purpose simulator that be used as a base to build simulations of various systems.
* ABIDES-Markets: Extension of ABIDES-Core to financial markets. Contains implementation of an exchange mimicking NASDAQ, stylised trading agents and configurations.
* ABIDES-Gym: Extra layer to wrap the simulator into an OpenAI Gym environment for reinforcement learning use. 2 ready to use trading environments available. Possibility to build other financial markets environments easily.

<p align="right">(<a href="#top">back to top</a>)</p>

## About this Fork

This project, **abides-hasufel**, extends ABIDES (Byrd & Balch, 2019), originally developed
at Georgia Tech and later maintained by J.P. Morgan Chase as
[abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public) (now archived).
This [fork](https://github.com/GabrieleDiCorato/abides-hasufel) modernizes the codebase with
updated dependencies, UV dependency management, performance fixes, and bug corrections.
See CHANGELOG.md for details.

<!-- CITING -->
## Citing ABIDES

[ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets](https://arxiv.org/pdf/2110.14771.pdf) or use
the following BibTeX:

```
@misc{amrouni2021abidesgym,
      title={ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets},
      author={Selim Amrouni and Aymeric Moulin and Jared Vann and Svitlana Vyetrenko and Tucker Balch and Manuela Veloso},
      year={2021},
      eprint={2110.14771},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```

[ABIDES: Towards High-Fidelity Market Simulation for AI Research](https://arxiv.org/abs/1904.12066)
or by using the following BibTeX:

```
@misc{byrd2019abides,
      title={ABIDES: Towards High-Fidelity Market Simulation for AI Research},
      author={David Byrd and Maria Hybinette and Tucker Hybinette Balch},
      year={2019},
      eprint={1904.12066},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Installation

1. Download the ABIDES source code, either directly from GitHub or with git:

    ```bash
    git clone https://github.com/GabrieleDiCorato/abides-hasufel
    cd abides-hasufel
    ```

    **Note:** The latest stable version is contained within the `main` branch.

2. **Option A: Quick Install with UV (Recommended)**

    ```bash
    # Install UV if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install ABIDES and all dependencies (runtime only)
    uv sync --no-dev

    # For development (includes dev dependencies, tests, docs)
    uv sync
    ```

3. **Option B: Using the install script**

    ```bash
    sh install.sh
    ```

4. **Option C: Traditional pip install**

    ```bash
    pip install -e .

    # For testing and docs support
    pip install -e .[test,docs]
    ```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage (regular)
Regular ABIDES simulations can be run either directly in python or through the command line

_For more examples, please refer to the notebooks in the `notebooks/` directory._

### Using the Declarative Config System (recommended):

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

config = SimulationBuilder().from_template("rmsc04").seed(0).build()
result = run_simulation(config)

print(result.metadata)           # seed, timing, tickers
print(result.markets["ABM"])     # per-ticker summary
```

`run_simulation()` compiles a fresh runtime dict internally, runs the
simulation, and returns an immutable `SimulationResult`.  The same
`SimulationConfig` can be passed to `run_simulation()` any number of times.

For the low-level path (direct Kernel access):

```python
from abides_markets.config_system import SimulationBuilder, compile
from abides_core import abides

config = SimulationBuilder().from_template("rmsc04").seed(0).build()
runtime = compile(config)       # fresh runtime dict — consumed once
end_state = abides.run(runtime)
```

See [`docs/ai/HASUFEL_CONFIG_SYSTEM.md`](docs/ai/HASUFEL_CONFIG_SYSTEM.md) for full
reference and [`notebooks/demo_Config_System.ipynb`](notebooks/demo_Config_System.ipynb)
for an interactive tutorial.

### Using procedural build_config (legacy):

```python
from abides_markets.configs import rmsc04
from abides_core import abides

config_state = rmsc04.build_config(seed = 0, end_time = '10:00:00')
end_state = abides.run(config_state)
```
<p align="right">(<a href="#top">back to top</a>)</p>

### Using the abides Command:

The config can be loaded and the simulation run using the `abides`
command in the terminal (from directory root):

```
$ abides abides-markets/abides_markets/configs/rmsc04.py --end_time "10:00:00"
```

The first argument is a path to a valid ABIDES configuration file.

Any further arguments are optional and can be used to overwrite any parameters
in the config file.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage (Gym)
ABIDES can also be run through a Gym interface using ABIDES-Gym environments.

```python
import gymnasium as gym
import abides_gym

env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
)

initial_state, info = env.reset(seed=0)
for i in range(5):
    state, reward, terminated, truncated, info = env.step(0)
```

## Default Available Markets Configurations

ABIDES currently has the following available background Market Simulation Configuration:

* RMSC03: 1 Exchange Agent, 1 POV Market Maker Agent, 100 Value Agents, 25 Momentum Agents, 5000 Noise Agents

* RMSC04: 1 Exchange Agent, 2 Market Maker Agents, 102 Value Agents, 12 Momentum Agents, 1000  Noise Agents

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License
Distributed under the BSD 3-Clause "New" or "Revised" License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
- ABIDES was originally developed by David Byrd and Tucker Balch at Georgia Tech: https://github.com/abides-sim/abides
- The J.P. Morgan Chase team ([Jared Vann](https://github.com/jaredvann), [Selim Amrouni](https://github.com/selimamrouni), [Aymeric Moulin](https://github.com/AymericCAMoulin)) maintained the [abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public) fork.
- This fork (abides-hasufel) is maintained by [Gabriele Di Corato](https://github.com/GabrieleDiCorato).

<p align="right">(<a href="#top">back to top</a>)</p>
