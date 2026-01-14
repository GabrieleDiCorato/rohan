# THESIS PROPOSAL

## R.O.H.A.N.: Risk Optimization with Heuristic Agent Network

# Preliminary Title

Evolutionary Market Microstructure: An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments

# Abstract

The validation of algorithmic trading strategies and market regulations traditionally relies on backtesting against historical data. However, this method suffers from a critical limitation: it is static. It fails to account for **"market impact**" (how the market reacts to the strategy itself) and "**emergent behavior**" (unpredictable feedback loops leading to volatility).

This thesis proposes an "Agentic Framework" that combines Agent-Based Modeling (ABM) with Large Language Models (LLMs). Instead of manually coding strategies, the system utilizes a "Meta-Agent" loop to autonomously generate, test, and refine Python-based trading logic within a realistic Limit Order Book (LOB) simulation.

The goal is to demonstrate that an AI system can iteratively "evolve" robust trading behaviors capable of surviving market shocks (e.g., flash crashes) without human intervention.

Results will be assessed numerically by evaluating the convergence of a strategy’s performance during the iteration.

#

# Problem Statement and Motivation

Financial Market Infrastructures (FMIs) like Borsa Italiana need robust tools to test software behavior under extreme conditions. Standard testing methods are insufficient:

* Static Backtesting: Cannot simulate how a new algorithm triggers a reaction from other market participants.
* Manual Iteration: Modifying code to handle edge cases (like high latency or liquidity drying up) is slow and labor-intensive.

There is a gap in the application of Generative AI for "Automated Mechanism Design." While LLMs can write code, they rarely have a closed-loop environment to test that code, observe the failure, and self-correct based on feedback.

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

#

# Methodology

The project will be executed in three phases:

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

# Tools & Technologies

* **Languages**
  Python (Core Simulation & AI orchestration), Java (Reference logic for Matching Engine if required).
* **Simulation Framework**
  Custom instance of the [ABIDES framework](https://arxiv.org/abs/1904.12066) (Agent-Based Interactive Discrete Event Simulation, by J.P. Morgan).
  This project is stale, with outdated dependencies: a fork with some upgrade work will be a necessary preliminary step.
* **AI Framework**
  Google ADK for the orchestration layer, with OpenRouter connections to various LLM providers;
* **Data Analysis**
  Pandas, Matplotlib (for visualizing order book dynamics).
* **Software Architecture & Development**
  Google Gemini 3 Pro, Github Copilot Pro, Visual Studio and Cursor IDEs will be instrumental in designing and implementing the project in the available time.

The project should be able to execute locally, with open-source code, and only require external resources to run LLM models.

# Expected Outcome

A functional pipeline that demonstrates "Self-Refining Code" in a financial context. This serves as a prototype for next-generation CI/CD pipelines in FinTech, where algorithms are automatically stress-tested and patched before human review.

Artifacts:

1. A Github project for the Agentic Framework;
2. A presentation to illustrate the problem, the methodologies, and results;

We highlight that, until recently, the project timeline and scope would be unfeasible for a single developer in the allotted time. The methodology of orchestrating agents to plan and execute coding and documentation tasks is a modern and important application of AI. We will document the coding methodology used in this project and provide feedback and insights.
