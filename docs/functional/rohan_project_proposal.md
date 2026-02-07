# THESIS PROPOSAL

## R.O.H.A.N.: Risk Optimization with Heuristic Agent Network


# Preliminary Title

Evolutionary Market Microstructure: An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments

# Abstract

The validation of algorithmic trading strategies and market regulations traditionally relies on backtesting against historical data. However, this method suffers from a critical limitation: it is static. It fails to account for *market impact* (how the market reacts to the strategy itself) and *emergent behavior* (unpredictable feedback loops leading to volatility).

This thesis proposes an "Agentic Framework" that combines Agent-Based Modeling (ABM) with Large Language Models (LLMs). Instead of manually coding strategies, the system utilizes a "Meta-Agent" loop to autonomously generate, test, and refine Python-based trading logic within a realistic Limit Order Book (LOB) simulation. The simulation itself is run by a large number of independent agents acting independently.

The goal is to demonstrate that an AI system can:

* understand complex market dynamics, by analyzing a large volume of information using dedicated tools;
* use the information to provide feedback on an algorithmic strategy’s implementation;
* iteratively "evolve" robust trading behaviors capable of surviving market shocks (e.g., flash crashes) without human intervention.


Results will be assessed numerically by evaluating the convergence of a strategy’s performance during the iteration.

#

# Problem Statement and Motivation

Financial Market Infrastructures (FMIs) like Borsa Italiana need robust tools to test software behavior under extreme conditions. Standard testing methods are insufficient:

* Static Backtesting: Cannot simulate how a new algorithm triggers a reaction from other market participants.
* Manual Iteration: Modifying code to handle edge cases (like high latency or liquidity drying up) is slow and labor-intensive.

There is a gap in the application of Generative AI for "Automated Mechanism Design." While LLMs can write code, they rarely have a closed-loop environment to test that code, observe the failure, and self-correct based on feedback.

Once the simulation environment is built and the strategy can be tested in a completely controlled environment, a few challenges remain:

* **Extracting Insights from Market Simulation Reports**
  A realistic market simulation generates a vast amount of data.
  Extracting useful information to understand each agent’s decisions and general market dynamics is a very demanding task for a human operator;
* **Translating Insights into Technical Specifications**
  Translating the market insights into technical directions to evolve the strategy’s implementation requires a complex array of specific knowledge;
* **Keeping track of the Strategy’s Evolution**
  Implementing and iterating the strategy evolution loop while keeping track of all the changes is complex. Convergence criteria and KPI might vary according to the strategy objectives.

The final result of an iteration is a piece of deterministic and clearly understandable Python code, not an intelligent AI agent able to trade autonomously on the market. This reflects a few key aspects:

* **Explainability of the solution and process design**
  The output strategy is completely explainable, modifiable by a human, and can be run in a Production environment without the risks involved in having an AI agent operate autonomously on the market.
* **Compatibility with existing testing frameworks**
  We aim at providing an additional utility on top of an institution’s current capabilities, so the strategy should be pluggable into a test environment or a trading system.

# Objectives

The primary objective is to build a Proof of Concept (PoC) pipeline that autonomously improves algorithmic performance. The focus is not on generating profit (*alpha*) but handling market stress and crisis scenarios without human intervention.

1. **The High-Frequency Trading Environment**
   Deploy a realistic Order Book simulation (leveraging the ABIDES framework) that supports multiple agent types (Market Makers, Noise Traders) and network latency.
2. **The Agentic Framework**
   Develop a control system using LangChain/LangGraph or Google ADK, where LLMs act as:
   * **Architect**: Generates Python code for a trading agent.
   * **Simulator**: Runs the code in the environment and captures execution KPIs over different scenarios.
     Execution KPIs are extracted from the detailed logging by a deterministic tool, avoiding LLM limitations.
   * **Analyst**: Interprets the KPIs (P\&L, drawdown, order rejection rates, limits, etc.) and drafts feedback for the Architect.
3. **The Experiment**
   Demonstrate the "Evolutionary" capability by subjecting the agent to a "Market Shock" scenario. We measure if the system successfully iterates its code to adapt to the new volatile regime. Convergence of KPIs and resilience will be key indicators: we want to show that the variance of the P\&L decreases over generations (i.e., the strategy becomes less erratic).

# Methodology

The project will be executed in three main phases, with optional improvements:

## Phase I: Simulation Infrastructure

Implementation of a discrete-event market simulator (custom ABIDES fork). This simulation will model the "Physics" of the market:

* A centralized Limit Order Book (LOB) matching engine;
* Stochastic "Noise" agents to provide background liquidity;
* A set of trading actors with specific, realistic strategies;
* Latency simulation to mimic real-world execution risks.

## Phase II: The Agentic Framework

Integration of LLMs via an agentic orchestration framework (Google ADK or langchain). The system will be designed as a feedback loop:

1. **Instruction**: "Create a Market Maker that provides liquidity”;
2. **Generation**: The LLM outputs a Python class \`Strategy\_V0.py\`.
   We will create a simplified Python base class to be extended in a framework-agnostic way, to avoid the complexity of the real trading API;
3. **Simulation**: The engine runs \`Strategy\_V0\` in the simulation environment;
4. **Evaluation**: The system detects the agent went bankrupt due to inventory risk;
5. **Refinement**: The Analyst agent prompts the Architect to "Add inventory skew logic";
6. **Iteration**: \`Strategy\_V1.py\` is generated and tested.

## Phase III: Evaluation & Stress Testing

The final system will be evaluated on:

* **Code Validity**: Can the system produce syntactically correct, executable Python code?
* **Adaptability**: Can the system modify a standard strategy to survive a sudden volatility spike (simulated Flash Crash)?
* **Interpretability**: Analysis of the "Reasoning Logs" (the conversation between Architect and Analyst).

## Phase IV (Optional): User Interface

A simple User Interface generated with the help of Large Language Models and existing front-end frameworks would involve little work and provide exceptional value to the presentation.

## Phase V (Optional): Reinforcement Learning

This environment is perfect to train and test high-frequency algo trading agents using reinforcement learning. The environment allows to simulate computation costs and latency, allowing to realistically evaluate an agent behaviour and compute an optimal threshold between performance and accuracy.

# Tools & Technologies

* **Languages**
  Python (Core Simulation & AI orchestration), Java (Reference logic for Matching Engine if required).
* **Simulation Framework**
  Custom instance of the [ABIDES framework](https://arxiv.org/abs/1904.12066) (Agent-Based Interactive Discrete Event Simulation, by J.P. Morgan).
  This project is stale, with outdated dependencies: a fork with some upgrade work will be a necessary preliminary step.
* **AI Framework**
  LangGraph, with OpenRouter connections to various LLM providers.
  At least a multi-modal model and a coding-focused model will be required;
* **Runtime Environment**
  Simulations involving LLM-generated strategies will run in isolated Docker environments.
* **Data Analysis**
  Pandas, Matplotlib (for visualizing order book dynamics).
* **Software Architecture & Development**
  Google Gemini 3 Pro, Github Copilot Pro, Visual Studio and Antigravity IDEs will be instrumental in designing and implementing the project in the available time.

The project should be able to execute locally, with open-source code, and only require external resources to run LLM models.

# Expected Outcome

A functional pipeline that demonstrates "Self-Refining Code" in a financial context. This serves as a prototype for next-generation CI/CD pipelines in FinTech, where algorithms are automatically stress-tested and patched before human review.

Artifacts:

1. A Github project for the Agentic Framework;
2. A presentation to illustrate the problem, the methodologies, and results;

We highlight that, until recently, the project timeline and scope would be unfeasible for a single developer in the allotted time. The methodology of orchestrating agents to plan and execute coding and documentation tasks is a modern and important application of AI. We will document the coding methodology used in this project and provide feedback and insights.

# Relevant Literature

Scientific literature regarding the ABIDES framework and market impact studies:

* \[2010Eisler\] [The price impact of order book events: market orders, limit orders and cancellations](https://arxiv.org/pdf/0904.0900)
* \[2019Byrd\] [ABIDES: TOWARDS HIGH-FIDELITY MARKET SIMULATION FOR AI RESEARCH](https://arxiv.org/pdf/1904.12066)
* \[2021Amrouni\] [ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets](https://arxiv.org/pdf/2110.14771)
* \[2021Coletta\] [Towards Realistic Market Simulations: a Generative Adversarial Networks Approach](https://arxiv.org/pdf/2110.13287)
* \[2025Cheridito\] [ABIDES-MARL: A Multi-Agent Reinforcement Learning Environment for Endogenous Price Formation and Execution in a Limit Order Book](https://arxiv.org/pdf/2511.02016)
* [ABIDES wiki (outdated)](https://github.com/abides-sim/abides/wiki)

Literature regarding agentic frameworks development and inner workings:

* \[2025Kaggle\] [5-Day AI Agents Intensive Course with Google](https://www.kaggle.com/learn-guide/5-day-agents) (5 whitepapers included)
* \[2025Gulli\] Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems
* \[2025Hua\] [Context Engineering 2.0: The Context of Context Engineering](https://arxiv.org/pdf/2510.26493)
* \[2025Weller\] [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_fP5hcU6-ls2eTnFl8oYGz9IqTZsXvKaxynB8msUWCa1wU3yB8yj0XFkoFTiIHKFAZQs5t)
*

# Resources

* [GitHub: ROHAN](https://github.com/GabrieleDiCorato/rohan)
  * Created extensive framework-agnostic configuration for market conditions and non-learning agents
  * Mapped the configuration to an abides-market simulation setup

* [GitHub: my ABIDES fork](https://github.com/GabrieleDiCorato/abides-jpmc-public)
  * Migrated to UV, all dependencies have been upgraded.
  * Added proper versioning to build artifacts, to import specific versions into ROHAN
  * A fundamental FinancialAgent implementation was missing (?\!) and the simulation was not running, I had to re-implement it.
  * Minor fixes on type safety
  * TODO: the testing framework is to be completely rewritten. Tests are present but at the moment they’re not testing anything of value. Non-regression tests compare an hardcoded GitHub commit ID with itself.

# Timeline

[ROHAN-GANT](https://docs.google.com/spreadsheets/d/1zriCFnc-90gq-CXr2VZ2byPTell6Y2gYP_zkr11wrwQ/edit?usp=sharing)

The work presented in this section is not necessarily to be executed sequentially. Most of the phases are decoupled by design and, once the communication interfaces are designed, can be worked on independently. Priority will be given to:

1. Preliminary feasibility study of the agentic trading simulation environment;
2. Design and implementation of the agentic framework designing the strategies.

What follows is a linear visualization of the project’s timeline.

## (Weeks 1-2) Phase I: Simulation Infrastructure

Market research has confirmed that [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) is the best open-source choice for our needs, despite numerous code quality and maintainability issues that will be addressed below.

This phase is dedicated to understanding, revamping, and configuring the ABIDES market simulation infrastructure:

1. (2 days) Read relevant scientific papers
2. (2days) The original repository is not maintained and outdated. It is archived, and pull requests are not accepted:
   1. Fork the repository into a [personal version](https://github.com/GabrieleDiCorato/abides-jpmc-public)
   2. Fix the ABIDES test suite
      Before upgrading all the dependencies required to modernize the library, we have to make sure our changes will not impact numerical results. The current test suite is broken and not exhaustive enough.
   3. Expand the test suite to cover numerical results and ensure future customizations will preserve the original simulation quality
3. (1 day) Upgrade the ABIDES dependencies to the latest standards, to allow for easier customization and integration in the ROHAN project.
4. (1 week) Configure and test our simulation market.
   Establish simulation parameters

## (Weeks 3-5) Phase II: The Agentic Framework

1. (2 days) Research to identify the KPIs that will be used to model the strategies’ performance and evaluate convergence of the improvement iteration
2. (2 days) Build upon ABIDES to extract the KPIs from the simulation environment’s logs
3. (1 day) Create a decoupling layer between the algorithmic strategy to be implemented by the LLM and the ABIDES API.
4. (1 week) Create an execution environment accepting an LLM-generated strategy dynamically:
   1. Validate the code generated by the LLM: linting and safety (limit possible imports)
   2. Create an isolated container where to safely execute this auto-generated code
   3. Load a strategy into an ABIDES trading agent and start the environment
   4. When the simulation completes, clean up and return the detailed KPIs extracted.
   5. Persist relevant information and logs in a dedicated database
5. (1 Week) Create the agentic framework using Google ADK
   1. Prompt engineering
   2. Context engineering
   3. Session and memory management
   4. Models selection

TODOS: document the behaviour of the analysis agent. To manage context, the prompt only contains a set of simulation KPIs. Cannot provide the full log (GBs). The agent will have a set of tools, grep-like, to explore the logs regarding specific times, actors, order types, or events. (recursive language model?)
