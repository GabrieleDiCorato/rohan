# ROHAN — Cost/Benefit Analysis

> **Version:** 1.1 — April 2026
> **Companion to:** [Target Architecture](target_architecture.md) (§1–16)
> **Audience:** Academic review panel, investment committee, CTO/CRO stakeholders.
>
> **Change log:**
> - v1.1 (Apr 2026): Dual-architecture LLM costing (current cyclic + target
>   DAG). Updated model pricing to GPT-5.4 / Claude Sonnet 4.6 families.
>   Added oversight sensitivity analysis (20/30/40%). Restated PoC maturity
>   (~80%). Added Savings Plan, egress, Redis justification, scale-to-zero
>   trade-off notes. Fixed Buy comparison comparables.

---

## 1. Analysis Framework

This analysis evaluates the target architecture along four dimensions:

| Dimension | Question Answered |
|---|---|
| **A. Total Cost of Ownership (TCO)** | What does it cost to build, run, and maintain? |
| **B. Quantitative Benefits** | What measurable value does it create? |
| **C. Qualitative Benefits** | What strategic and regulatory advantages does it provide? |
| **D. Risk-Adjusted Assessment** | What could go wrong and how does it affect the equation? |

Costs are broken into **build** (one-time development) and **run** (recurring
monthly/annual). Benefits are broken into **direct** (cost savings, revenue
protection) and **indirect** (risk reduction, compliance, strategic).

All estimates use AWS pricing (eu-west-1) as of April 2026. GCP/Azure costs
are comparable (±10%). LLM costs use OpenRouter published pricing.

---

## 2. Total Cost of Ownership

### 2.1 Cloud Infrastructure — Monthly Run Cost

#### Compute

| Component | Specification | Unit Cost | Quantity | Monthly Cost |
|---|---|---|---|---|
| **API Gateway** | ECS Fargate, 0.5 vCPU / 1 GB | $0.025/hr | 2 tasks × 730 hrs | **$36** |
| **Orchestrator** | ECS Fargate, 1 vCPU / 2 GB | $0.050/hr | 2 tasks × 730 hrs | **$73** |
| **Simulation Workers** | ECS Fargate, 2 vCPU / 4 GB | $0.100/hr | 4 tasks avg × 730 hrs | **$292** |
| **Worker auto-scale burst** | Same spec, on-demand | $0.100/hr | ~200 burst hrs/mo | **$20** |
| | | | **Compute subtotal** | **$421/mo** |

*Workers dominate. Scaling from 4 → 16 baseline workers would raise compute to ~$1,200/mo. Right-sizing workers to actual utilization (scale-to-zero during off-hours) can reduce baseline to ~$250/mo, though Fargate cold starts of 30–60 seconds should be expected after scale-to-zero periods — acceptable for scheduled runs but may affect on-demand UX.*

*1-year Compute Savings Plans reduce Fargate costs by ~20–30% for steady-state
workloads (AWS advertises up to 66% but that requires 3-year commitment and
high sustained utilization). RDS Reserved Instances save 30–40% vs. on-demand.
The table above uses on-demand pricing as the conservative baseline.*

#### Data

| Component | Specification | Unit Cost | Quantity | Monthly Cost |
|---|---|---|---|---|
| **PostgreSQL** | RDS db.r6g.large, Multi-AZ | $0.48/hr | 730 hrs | **$350** |
| **RDS Storage** | gp3, 100 GB | $0.115/GB/mo | 100 GB | **$12** |
| **Redis** | ElastiCache t3.small | $0.034/hr | 730 hrs | **$25** |
| **S3 Object Storage** | Standard | $0.023/GB/mo | 50 GB (growing) | **$1** |
| **S3 Request costs** | PUT/GET | ~$0.005/1000 | ~50K req/mo | **$0.25** |
| | | | **Data subtotal** | **$388/mo** |

*PostgreSQL is the second-largest cost. For lower-volume deployments, db.t4g.medium ($0.12/hr → ~$88/mo) in single-AZ is viable with automated backups.*

*Redis is budgeted for the target architecture's rate limiting, result caching,
and real-time job status tracking (API Gateway → worker coordination). The PoC
does not currently use Redis; this line item applies only to the cloud
deployment profile.*

#### Network & Ancillary

| Component | Specification | Monthly Cost |
|---|---|---|
| **ALB** | Application Load Balancer | **$22** |
| **NAT Gateway** | For private subnet egress | **$35** |
| **CloudWatch Logs** | ~5 GB/mo | **$3** |
| **Secrets Manager** | 5 secrets | **$2** |
| **SES** (email) | ~500 emails/mo | **$0.50** |
| **Data transfer (egress)** | ~100 GB/mo at $0.09/GB | **$9** |
| | **Network subtotal** | **$72/mo** |

#### Infrastructure Total

| Profile | Monthly | Annual |
|---|---|---|
| **Production baseline** (4 workers, Multi-AZ DB) | **$872/mo** | **$10,464/yr** |
| **Production w/ 1-yr Savings Plan** (same spec, committed) | **$640–$700/mo** | **$7,700–$8,400/yr** |
| **Cost-optimized** (2 workers, single-AZ DB, scale-to-zero) | **$450/mo** | **$5,400/yr** |
| **High-throughput** (16 workers, larger DB) | **$2,100/mo** | **$25,200/yr** |

---

### 2.2 LLM API Costs — Per Validation Run

LLM costs depend on the graph topology. The current PoC uses a **cyclic
refinement loop** (Writer → Validate → Execute → Explain → Aggregate → Writer,
up to `max_iterations` iterations). The target architecture replaces this with
a **linear DAG** (Analyzer → Planner → Execute → Explain → Aggregate → Report)
with a fixed number of LLM calls per run. Both are costed below.

In both architectures, **scoring is fully deterministic** — no LLM in the
scoring loop — which caps the variable-cost component.

> **Note on model pricing:** Model names and prices move quickly. The table
> below uses April 2026 list prices from OpenAI and Anthropic. By the time
> this document is reviewed, newer models may offer better price/performance.
> The model-agnostic factory pattern in the codebase allows switching providers
> via configuration.

#### A. Current Architecture (Cyclic Refinement Loop)

The current PoC runs up to `max_iterations` (default 5) refinement cycles.
Each iteration invokes Writer + Validator (no LLM) + Explainer + Aggregator.
The Planner runs once before the loop.

| Node | Model Tier | Input Tokens (est.) | Output Tokens (est.) | Calls per Iteration |
|---|---|---|---|---|
| **Planner** (once) | Flagship | ~3,000 | ~2,000 | 1–3 (ReAct steps) |
| **Writer** | Flagship | ~5,000 (goal + feedback) | ~2,000 (strategy code) | 1 |
| **Explainer** (per scenario) | Flagship | ~6,000 (rich analysis + tools) | ~2,000 (explanation) | 3–8 tool calls × N scenarios |
| **Aggregator** | Flagship | ~8,000 (all explanations + scores) | ~2,000 (synthesis + feedback) | 1 |

**Cost per run (3 scenarios, 3 iterations, GPT-5.4 at $2.50/$15.00 per 1M tokens):**

| Component | Input Tokens | Output Tokens | Cost |
|---|---|---|---|
| Planner (1×) | ~6,000 | ~4,000 | $0.08 |
| Writer (3×) | ~15,000 | ~6,000 | $0.13 |
| Explainer (3 scenarios × 3 iters × 5 calls) | ~270,000 | ~90,000 | $2.03 |
| Aggregator (3×) | ~24,000 | ~6,000 | $0.15 |
| **Total (3 iterations, 3 scenarios)** | **~315,000** | **~106,000** | **~$2.38** |

With 5 iterations: **~$3.50–$4.00/run.** With 12 scenarios × 3 iterations:
**~$8–$10/run.** The cyclic architecture makes per-run LLM cost
**proportional to iterations × scenarios**, which is harder to budget.

#### B. Target Architecture (Linear DAG)

The target DAG executes each node once. No iteration loop.

| Node | Model Tier | Input Tokens (est.) | Output Tokens (est.) | Calls per Run |
|---|---|---|---|---|
| **Analyzer** | Flagship (GPT-5.4 / Claude Sonnet 4.6) | ~4,000 (strategy code + metadata) | ~1,500 (StrategyProfile) | 1 |
| **Planner** | Flagship | ~3,000 (profile + scenario registry) | ~2,000 (adversarial plan) | 1–3 (ReAct steps) |
| **Explainer** (per scenario) | Flagship | ~6,000 (rich analysis + tools) | ~2,000 (explanation) | 3–8 tool calls × N scenarios |
| **Aggregator** | Flagship | ~8,000 (all explanations + scores) | ~2,000 (synthesis) | 1 |
| **ReportBuilder** | Lightweight (GPT-5.4 nano / Haiku 4.5) | ~3,000 (structured data) | ~1,500 (narrative) | 1 |

**Cost per run (12 scenarios, GPT-5.4 at $2.50/$15.00 per 1M tokens):**

| Model | Input Price / 1M tokens | Output Price / 1M tokens | Est. Input Tokens | Est. Output Tokens | **Cost** |
|---|---|---|---|---|---|
| **GPT-5.4** (flagship nodes) | $2.50 | $15.00 | ~120,000 | ~45,000 | **$0.98** |
| **GPT-5.4 nano** (report) | $0.20 | $1.25 | ~3,000 | ~1,500 | **$0.003** |
| | | | | **Per-run LLM total** | **~$1.00** |

| Alternative Model | Per-Run Cost |
|---|---|
| Claude Sonnet 4.6 (via OpenRouter) | ~$0.96 (similar; $3/$15 pricing) |
| Claude Haiku 4.5 (budget flagship) | ~$0.35 |
| DeepSeek V3 (budget) | ~$0.10 |

The linear DAG makes per-run cost **fixed and predictable** — no unbounded
iteration.

#### Monthly LLM Cost by Usage Volume

| Usage Profile | Runs/Month | Current Arch. (3 iter avg) | Target Arch. (linear DAG) |
|---|---|---|---|
| **Light** (1 team, on-demand only) | 50 | **$119** | **$50** |
| **Medium** (3 teams, on-demand + weekly scheduled) | 200 | **$476** | **$200** |
| **Heavy** (10+ teams, daily scheduled + CI) | 1,000 | **$2,380** | **$1,000** |

*LLM costs remain minor relative to infrastructure in both architectures.
Moving to the linear DAG approximately halves LLM spend and — more
importantly — makes it fully predictable. Scoring is deterministic in both
architectures (no LLM in the scoring loop).*

*The remainder of this CBA uses **target architecture (linear DAG) costs** for
the ROI calculations, since the DAG refactor is included in the build scope
(§2.3, "Orchestrator refactor" line item). Current-architecture costs are
higher but do not change the directional conclusions.*

---

### 2.3 Development Cost — Build Phase

#### Effort Estimate by Component

| Component | Description | Complexity | Est. Effort (person-weeks) |
|---|---|---|---|
| **FastAPI service** | REST API, auth integration, RBAC | Medium | 3–4 |
| **Orchestrator refactor** | Cyclic graph → linear DAG, new nodes | Medium | 2–3 |
| **Analyzer node** | LLM code comprehension, StrategyProfile model | Medium | 2 |
| **Strategy adapter layer** | AST analysis, discrete-time translation | High | 4–6 |
| **Scenario registry** | DB schema, admin CRUD, versioning, approval workflow | Medium | 3–4 |
| **Report builder** | Multi-format rendering (PDF, JSON, email) | Medium | 3–4 |
| **Scheduled validation** | Cron integration, notification pipeline | Low | 1–2 |
| **CI webhook trigger** | Webhook endpoint, strategy version detection | Low | 1 |
| **Observability** | OpenTelemetry instrumentation, dashboards, alerts | Medium | 2–3 |
| **DB schema migration** | PoC schema → target schema, data migration | Medium | 2 |
| **Containerization** | Dockerfiles, ECS/EKS task definitions, networking | Medium | 2–3 |
| **Infrastructure as Code** | Terraform/Pulumi for full cloud deployment | Medium | 3–4 |
| **Testing & QA** | Integration tests, load tests, security audit | Medium | 3–4 |
| **Documentation** | API docs, onboarding guide, runbooks | Low | 1–2 |
| | | **Total** | **32–45 person-weeks** |

#### Build Cost Scenarios

| Team Size | Duration | Effective Rate | Total Build Cost |
|---|---|---|---|
| 1 senior engineer | 8–11 months | €600–900/day | **€96K–198K** |
| 2 engineers (1 senior + 1 mid) | 4–6 months | €800–1,400/day combined | **€64K–168K** |
| 3 engineers (typical startup) | 3–4 months | €1,200–2,000/day combined | **€72K–160K** |

*The PoC already implements **~80% of the core algorithm and data pipeline logic**
(simulation engine, scoring, 8 explainer tools, planner, sandbox, persistence
with 5-table SQLAlchemy schema, LangGraph orchestration). The build phase is
concentrated on service decomposition, deployment infrastructure, and new nodes
(Analyzer, ReportBuilder) — not core R&D.*

---

### 2.4 Ongoing Operational Cost

| Item | Monthly Cost | Notes |
|---|---|---|
| **Cloud infrastructure** | $450–$2,100 | Per §2.1 profile |
| **LLM API** | $50–$1,000 | Per §2.2 volume (target arch.) |
| **Identity provider** | $0–$300 | Auth0 free tier covers <1000 users; enterprise negotiable |
| **Monitoring** (Grafana Cloud) | $0–$50 | Free tier likely sufficient; $50 for pro features |
| **DevOps / SRE** (part-time) | €2,000–€5,000 | 0.25–0.5 FTE for platform maintenance |
| **LLM model upgrades** | €500–€1,000 | Quarterly prompt/model refresh and regression testing |
| | **Monthly operational total** | **€3,000–€8,500** |
| | **Annual operational total** | **€36K–€102K** |

---

### 2.5 TCO Summary (3-Year Horizon)

| Cost Category | Year 1 | Year 2 | Year 3 | 3-Year Total |
|---|---|---|---|---|
| **Build** (one-time) | €80K–€170K | — | — | €80K–€170K |
| **Infrastructure** (annual) | €5.4K–€25.2K | €5.4K–€25.2K | €5.4K–€25.2K | €16K–€76K |
| **LLM API** (annual) | €0.5K–€9K | €0.5K–€9K | €0.5K–€9K | €1.5K–€27K |
| **Operations** (annual) | €36K–€102K | €36K–€102K | €36K–€102K | €108K–€306K |
| | | | **3-Year TCO** | **€206K–€579K** |

| Profile | Realistic 3Y TCO |
|---|---|
| **Lean startup** (1 eng, cost-optimized infra, light usage) | **~€250K** |
| **Mid-scale** (2 eng build, prod infra, medium usage) | **~€380K** |
| **Enterprise** (3 eng build, high-throughput, heavy usage) | **~€550K** |

---

## 3. Quantitative Benefits

### 3.1 Baseline: Cost of Current Manual Validation

Model validation in quantitative finance is labor-intensive. Industry benchmarks:

| Activity | Manual Effort | Cost (Senior Quant Analyst, ~€150/hr) |
|---|---|---|
| **Initial strategy validation** | 2–5 days per strategy | €2,400–€6,000 |
| **Quarterly re-validation** | 1–2 days per strategy | €1,200–€2,400 |
| **Regulatory stress test design** | 1–2 days per event | €1,200–€2,400 |
| **Report writing & review** | 1–3 days per report | €1,200–€3,600 |
| **Audit response** (reproducibility challenge) | 2–5 days per event | €2,400–€6,000 |

*Sources: McKinsey (2023) model risk management benchmarks, EY model validation
survey, industry interviews.*

#### Annual Manual Validation Cost (per strategy portfolio)

| Portfolio Size | Initial + Recurring Annual Cost |
|---|---|
| **10 strategies** (small desk) | €60K–€120K/yr |
| **50 strategies** (mid-size firm) | €240K–€480K/yr |
| **200 strategies** (large quant firm) | €800K–€1.5M/yr |

### 3.2 Automated Validation Cost with ROHAN

Automation reduces but does not eliminate human oversight. Analysts review
AI-generated reports, govern the scenario registry, and validate flagged edge
cases. The residual human effort depends on organizational maturity and
regulatory posture.

Industry benchmarks for AI-augmented professional workflows in regulated
financial services (Deloitte MRM survey 2024, McKinsey 2023 risk management
benchmarks, OCC/Fed SR 11-7 implementation studies) suggest:

- **20% residual** — optimistic; assumes high trust in AI outputs, mature
  tooling, streamlined governance. Achievable for re-validation runs on
  previously approved strategies.
- **30% residual** — base case; reflects typical AI-augmented workflows where
  human review is policy-mandated. Aligns with reported efficiency gains in
  comparable regulatory compliance automation.
- **40% residual** — conservative; reflects early adoption, cautious compliance
  culture, or strategies requiring significant manual interpretation.

#### Base Case (30% oversight)

| Portfolio Size | Runs/Year | LLM Cost | Infra Cost | Quant Analyst Oversight (30% of manual) | **Total** |
|---|---|---|---|---|---|
| **10 strategies** | 250 | €200 | €5.4K–€10K | €18K–€36K | **€24K–€46K** |
| **50 strategies** | 1,000 | €1K | €10K–€25K | €72K–€144K | **€83K–€170K** |
| **200 strategies** | 4,000 | €4K | €25K–€50K | €240K–€450K | **€269K–€504K** |

#### Sensitivity by Oversight Rate (50-strategy portfolio)

| Oversight Rate | Analyst Cost | LLM + Infra | Total ROHAN Cost | Annual Savings vs. Manual |
|---|---|---|---|---|
| **20%** (optimistic) | €48K–€96K | €11K–€26K | €59K–€122K | **€118K–€421K** |
| **30%** (base case) | €72K–€144K | €11K–€26K | €83K–€170K | **€70K–€397K** |
| **40%** (conservative) | €96K–€192K | €11K–€26K | €107K–€218K | **€22K–€373K** |

*At 30% oversight (base case), savings remain meaningful at all portfolio sizes.
At 40% oversight, the 10-strategy portfolio becomes marginal (€14K–€62K savings
vs. €80K–€170K build cost), requiring multi-year payback. The dominant variable
is oversight rate, not infrastructure or LLM cost.*

### 3.3 Direct Savings (Base Case — 30% Oversight)

| Portfolio Size | Manual Cost | ROHAN Cost (30%) | **Annual Savings** | **Payback Period** |
|---|---|---|---|---|
| **10 strategies** | €60K–€120K | €24K–€46K | **€14K–€96K** | 2–5 years |
| **50 strategies** | €240K–€480K | €83K–€170K | **€70K–€397K** | 5–18 months |
| **200 strategies** | €800K–€1.5M | €269K–€504K | **€296K–€1M** | 2–7 months |

*At the base case (30% oversight), breakeven on 10 strategies requires 2+ years.
For ≥50 strategies, the platform pays for itself within the first 18 months
including build costs. At the optimistic 20% oversight rate, payback on 50
strategies is 5–11 months. The investment thesis is robust at ≥30 strategies
across all oversight assumptions.*

### 3.4 Efficiency Multiplier

| Metric | Manual | ROHAN | Improvement |
|---|---|---|---|
| **Time to validate** (initial) | 2–5 days | 15–45 minutes | **10–50×** |
| **Time to re-validate** (recurring) | 1–2 days | 15–45 minutes (automated, scheduled) | **30–100×** (zero analyst time if scheduled) |
| **Scenarios per validation** | 3–5 (human-designed) | 8–15 (mandatory + AI adversarial) | **2–5×** coverage |
| **Reproducibility** | Difficult (manual setup) | Guaranteed (deterministic seeds) | Binary improvement |
| **Audit response time** | 2–5 days (reconstruct analysis) | Instant (retrieve immutable report) | **>100×** |
| **Report turnaround** | 1–3 days | Seconds (auto-generated) | **>100×** |

---

## 4. Qualitative Benefits

### 4.1 Regulatory Compliance

| Benefit | Regulatory Driver | Impact |
|---|---|---|
| **Demonstrable stress testing** | MiFID II Art. 17(1) | Mandatory scenario registry provides auditable evidence of systematic stress testing. Without it, firms rely on ad-hoc analyst judgment — a known audit finding. |
| **Independent model validation** | SR 11-7 §§5-8 | AI-inferred intent cross-validated against declared objective provides an independent assessment layer. The system cannot be "gamed" by the strategy developer who also writes the validation report. |
| **Ongoing monitoring** | SR 11-7 §9 | Scheduled re-validation with automated delta comparison replaces manual quarterly reviews that are often delayed or deprioritized. |
| **Full audit trail** | SR 11-7 §11 | Immutable reports, versioned scenarios, and access logging provide end-to-end traceability. Every finding can be traced back to specific simulation data. |
| **Model inventory** | EBA GL 2017/11 | Strategy registry with version history and validation status provides a live model inventory. |

**Regulatory risk avoided:** A single regulatory finding related to inadequate model validation can result in:
- Required remediation programs (€500K–€2M for mid-size firms)
- Increased capital requirements under Pillar 2
- Restrictions on algorithmic trading activity (revenue impact)
- Reputational damage

*The platform doesn't eliminate regulatory risk, but it provides systematic evidence of compliance that materially reduces the probability and severity of findings.*

### 4.2 Risk Management Quality

| Benefit | Description |
|---|---|
| **Deeper coverage** | AI-designed adversarial scenarios target strategy-specific weaknesses that human analysts may not consider. The planner has access to the full template library and can compose novel scenario combinations. |
| **Forensic explainability** | The Explainer agent's 8 investigation tools can drill into individual fills, order lifecycles, and L2 snapshots. This level of detail is impractical to produce manually for every validation. |
| **Consistency** | Every strategy is evaluated against the same mandatory scenario set with the same scoring methodology. Eliminates analyst-to-analyst variance in validation quality. |
| **Speed enables iteration** | 15-minute turnaround allows quant developers to validate iteratively during development — catching issues before deployment rather than after. This shifts risk management left in the development lifecycle. |
| **Cross-strategy comparison** | Standardized scoring enables meaningful comparison across strategies, informing portfolio allocation decisions with quantitative risk data. |

### 4.3 Strategic & Organizational

| Benefit | Description |
|---|---|
| **Scalability** | Headcount-independent scaling. Adding 50 strategies to the portfolio doesn't require hiring 50% more validation analysts. |
| **Knowledge preservation** | Validation logic is codified in scenarios, scoring formulas, and prompts — not trapped in individual analysts' heads. |
| **Faster time-to-market** | Strategies reach production faster because validation is no longer a multi-day bottleneck. For alpha-decaying strategies, speed has direct P&L impact. |
| **Data asset** | Historical validation runs across all strategies build a dataset of strategy behaviors under stress. This dataset has value for research, portfolio construction, and risk modeling. |

---

## 5. Risk-Adjusted Assessment

### 5.1 Key Risks and Mitigations

| # | Risk | Probability | Impact | Mitigation | Residual Risk |
|---|---|---|---|---|---|
| 1 | **Adapter fidelity** — many real-time strategies cannot be faithfully translated to discrete-time | Medium | High — reduces addressable market | Protocol-native path is zero-fidelity-loss; adapter builds incrementally from real strategy patterns | Medium |
| 2 | **Simulation fidelity** — ABIDES is a stylized model, not a real exchange; users over-trust results | Medium | Medium — credibility risk | Reports include explicit model limitations section; scoring is relative (strategy vs. baseline), not absolute | Low |
| 3 | **LLM reliability** — Analyzer/Planner/Explainer produce incorrect or misleading analysis | Medium | Medium — report quality | Three-tier fallbacks (ReAct → structured → heuristic); deterministic scoring is LLM-free; human review of reports is expected | Low |
| 4 | **LLM cost inflation** — model providers increase pricing | Low | Low — LLM is <5% of TCO | Model-agnostic factory pattern; can switch providers in config; commodity models (DeepSeek) as fallback | Very Low |
| 5 | **Regulatory interpretation** — SR 11-7 / MiFID II requirements are interpreted more strictly than expected | Low | Medium — may require additional features | Scenario registry governance and report immutability are designed to the strictest reasonable interpretation | Low |
| 6 | **Single-ticker limitation** — production strategies trade multiple assets; ABIDES supports one ticker | High | High — limits applicability | Document as known limitation; plan multi-ticker hasufel support; some strategies can be decomposed per-ticker | High |
| 7 | **Build overrun** — development takes longer or costs more than estimated | Medium | Medium — delayed time-to-value | ~80% of core logic exists in PoC; phased delivery (MVP → full features); regular milestones | Low |
| 8 | **Adoption resistance** — quant teams perceive AI validation as unreliable or threatening | Medium | Medium — underutilization | Position as augmentation (analyst reviews report, doesn't produce it); demonstrate forensic depth exceeding manual capability | Medium |

### 5.2 Risk-Adjusted NPV Sensitivity

The payback and savings estimates from §3.3 assume full adoption and 30%
oversight (base case). Adjusted for adoption risk:

| Scenario | Adoption Rate | Effective Savings (50-strategy portfolio, 30% oversight) | Payback (including build) |
|---|---|---|---|
| **Optimistic** | 90% | €63K–€357K/yr | 5–12 months |
| **Base case** | 70% | €49K–€278K/yr | 7–18 months |
| **Conservative** | 50% | €35K–€199K/yr | 10–24 months |
| **Pessimistic** | 30% | €21K–€119K/yr | 18–40 months |

*Even in the pessimistic scenario (30% adoption, 30% oversight, highest cost
estimates), the platform achieves payback within ~3.5 years on a 50-strategy
portfolio. At 70% adoption, payback is within 18 months. The dominant variables
are adoption rate and oversight rate, not infrastructure or LLM cost.*

---

## 6. Comparison Matrix: Build vs. Buy vs. Manual

| Dimension | Manual (Status Quo) | Build (ROHAN Target) | Buy (Commercial Vendor) |
|---|---|---|---|
| **Annual cost (50 strategies)** | €240K–€480K | €83K–€170K | €150K–€400K (enterprise license) |
| **Customization** | Full (but expensive) | Full (own codebase) | Limited to vendor roadmap |
| **Scenario design** | Human judgment only | AI + mandatory registry | Vendor-defined stress tests |
| **Simulation fidelity** | Depends on vendor/internal tools | ABIDES full LOB simulation | Typically simplified models |
| **Forensic depth** | Depends on analyst skill | 8-tool investigation suite | Typically aggregate metrics only |
| **Explainability** | Analyst writes narrative | AI-generated with code-level specificity | Usually scores without explanation |
| **Audit trail** | Manual document management | Automated, immutable, version-controlled | Vendor-dependent |
| **Time-to-validate** | Days | Minutes | Hours |
| **Vendor lock-in** | None | None (open-source stack) | High |
| **IP control** | Full | Full (strategy code stays internal) | Strategy code shared with vendor |
| **Regulatory acceptance** | Established | Must demonstrate rigor | Vendor provides compliance materials |

*The “Buy” column aggregates capabilities from enterprise model risk management
platforms (SAS Model Risk Management, Moody’s Analytics MRM, vendor-internal
MRM teams) and algo-trading-specific validation suites. Pricing ranges reflect
published list prices and reported enterprise deals (€150K–€400K/yr for 50+
models is consistent with Chartis Research MRM vendor rankings, 2024–2025).
Bloomberg MARS and Axioma Risk are portfolio risk systems, not model validation
platforms — they are excluded from this comparison. Key differentiator: ROHAN
provides AI-driven adversarial scenario design and forensic-level
explainability. Commercial MRM vendors typically offer predefined stress tests
with aggregate metrics.*

*The "IP control" row is critical for quant firms. Sharing strategy source code
with a third-party vendor is often unacceptable. ROHAN keeps all data internal.*

---

## 7. Investment Recommendation

### 7.1 Decision Criteria Assessment

| Criterion | Score (1–5) | Rationale |
|---|---|---|
| **Strategic alignment** | 5 | Directly addresses regulatory obligations and risk management quality |
| **Financial return** | 4 | Positive ROI at ≥30 strategies; sub-year payback at ≥50 strategies |
| **Technical feasibility** | 4 | ~80% of core logic proven in PoC; remaining is service decomposition and new nodes |
| **Execution risk** | 3 | Adapter fidelity and single-ticker limitation are material risks |
| **Competitive differentiation** | 5 | No equivalent product combines AI adversarial design + LOB simulation + forensic explainability |
| **Time-to-value** | 4 | MVP (protocol-native path, on-demand trigger, interactive reports) achievable in 3–4 months |

### 7.2 Recommended Phased Approach

| Phase | Scope | Duration | Delivers |
|---|---|---|---|
| **Phase 1: MVP** | Protocol-native ingestion, linear DAG (Analyzer → Planner → Execute → Explain → Report), on-demand trigger, interactive UI + JSON export, PostgreSQL, single-server deployment | 3–4 months | Core validation capability; usable for internal teams |
| **Phase 2: Production** | FastAPI service, containerization, cloud deployment (ECS/Cloud Run), PDF reports, scheduled validation, RBAC, observability | 2–3 months | Production-grade platform; auditable |
| **Phase 3: Enterprise** | Strategy adapter layer, CI webhook trigger, scenario registry governance (approval workflows), email notifications, cross-run comparison, multi-format exports | 2–3 months | Full target architecture per spec |
| **Phase 4: Evolution** | Multi-ticker support (hasufel roadmap), historical market data integration, portfolio-level analysis, API marketplace | Ongoing | Market expansion |

### 7.3 Key Takeaways

1. **The DAG simplification is a cost and predictability advantage.** Moving from the current cyclic refinement loop to a linear DAG makes LLM spend predictable (~$1.00/run vs. ~$2.40–$4.00/run in the current architecture). The current architecture’s costs are bounded by `max_iterations` but still variable. Both architectures keep LLM costs below 5% of TCO.

2. **Infrastructure is the majority cost, not LLM.** At medium usage, LLM API costs are ~2% of TCO. Building cost-efficiently (right-size workers, scale-to-zero) has more impact than model selection.

3. **The PoC de-risks the build.** ~80% of the core algorithm and data pipeline logic (simulation engine, scoring system, explainer tools, planner, sandbox, persistence, LangGraph orchestration) is already built and tested. The target architecture is primarily a deployment, integration, and new-node effort — not a fundamental R&D challenge.

4. **Payback depends on portfolio size and human oversight rate.** Below 20 strategies, the lean deployment makes sense but payback extends to 2+ years at the 30% oversight base case. Above 50, the platform is a clear cost saver. Above 100, the savings fund the platform multiple times over. Sensitivity to the oversight assumption is significant: at 20% oversight, savings are ~40% higher than at 30%.

5. **The single-ticker limitation is the biggest product risk.** It constrains the addressable market to single-instrument strategies. Multi-ticker support should be prioritized on the hasufel roadmap.

6. **Regulatory compliance is the non-negotiable value.** Even if the quantitative savings alone don't justify the build, the systematic compliance evidence (immutable reports, versioned scenarios, deterministic reproducibility) addresses a real regulatory gap that manual processes cannot close at scale.

---

## Appendix A: Pricing Assumptions

| Item | Source | Value Used |
|---|---|---|
| ECS Fargate (eu-west-1) | AWS pricing page, Apr 2026 | $0.04048/vCPU/hr + $0.004445/GB/hr |
| RDS PostgreSQL db.r6g.large Multi-AZ | AWS pricing page | $0.48/hr |
| ElastiCache t3.small | AWS pricing page | $0.034/hr |
| S3 Standard | AWS pricing page | $0.023/GB/mo |
| GPT-5.4 (input/output) | OpenAI pricing, Apr 2026 | $2.50/$15.00 per 1M tokens |
| GPT-5.4 nano (input/output) | OpenAI pricing | $0.20/$1.25 per 1M tokens |
| Claude Sonnet 4.6 (input/output) | Anthropic pricing, Apr 2026; same via OpenRouter | $3.00/$15.00 per 1M tokens |
| Claude Haiku 4.5 (input/output) | Anthropic pricing | $1.00/$5.00 per 1M tokens |
| Data transfer (egress) | AWS pricing page | $0.09/GB (first 10 TB/mo) |
| Senior Quant Analyst (EU) | Robert Half 2025 salary guide, loaded | €150/hr |
| Senior Software Engineer (EU) | Robert Half 2025 salary guide, loaded / contractor rate | €600–900/day |

## Appendix B: Token Estimation Methodology

Token estimates are based on PoC measurements:

| Data | Measured Tokens | Method |
|---|---|---|
| Strategy code (typical) | 800–2,000 | `tiktoken` on PoC-generated strategies |
| `RichAnalysisBundle` JSON (per scenario) | 3,000–5,000 | `.model_dump_json()` → `tiktoken` |
| System prompts (Explainer) | ~1,500 | Direct measurement from `prompts.py` |
| Tool call/response (average) | ~400 per round-trip | Measured from Explainer ReAct traces |
| `ScenarioExplanation` (output) | ~800 | Average structured output size |

Estimates include a 1.3× buffer for prompt engineering overhead (system
instructions, formatting, safety preambles).

**Context accumulation note:** In the current cyclic architecture, later
iterations carry accumulated context (previous explanations, feedback) that
increases prompt size per iteration. The estimates above use per-iteration
averages; the first iteration is cheaper and the last is more expensive. The
1.3× buffer partially accounts for this but may underestimate by ~10–15% for
5-iteration runs with 12 scenarios. The target linear DAG does not have this
problem (each node runs once).
