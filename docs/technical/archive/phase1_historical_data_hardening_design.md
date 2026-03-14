# Phase 1 Historical Data Hardening - Detailed Technical Design

## 1. Scope and Objective

This design executes Phase 1 of scenario customizability by hardening and completing the historical data pipeline around ExternalDataOracle.

Primary goals:
- eliminate correctness risks in normalization (timezone and unit handling)
- enforce typed and fail-fast oracle/provider configuration
- implement true provider dispatch for CSV, database, and API backends
- keep one normalization pipeline for all providers
- expand tests to cover failure modes and integration behavior

Out of scope for this phase:
- Phase 2 plugin architecture
- UI monolith decomposition
- migration scripts (explicitly not required)

## 2. Current Gaps (Baseline)

Observed implementation gaps against Phase 1 requirements:
- timezone-aware timestamps are converted via tz_convert(None), which can shift wall-clock market times
- provider_type exists in config but mapper always instantiates CSV provider
- invalid interpolation silently falls back to forward-fill
- normalization assumes values are already cents
- only CSV provider exists; DB/API provider abstraction is incomplete
- tests cover happy paths but not key failure contracts

## 3. Design Decisions

### 3.1 Timezone Semantics

Decision:
- internal normalized series will remain timezone-naive because ABIDES expects naive nanosecond timestamps
- conversion to naive must preserve exchange-local wall-clock timestamps

Rules:
- incoming tz-aware index is first converted to a configured exchange timezone, then tz info is removed
- default exchange timezone for this phase: America/New_York
- if index is tz-aware and timezone policy is invalid or missing, raise ValueError

Rationale:
- preserves 09:30 -> 09:30 semantics for session slicing and avoids hidden UTC shifts

### 3.2 Explicit Price Unit Contract

Decision:
- normalization receives explicit input unit metadata
- accepted units: CENTS, DOLLARS
- unknown units are rejected

Rules:
- DOLLARS: multiply by 100 and round to int
- CENTS: round to int directly
- validation runs after unit conversion and optional recentering

Rationale:
- removes implicit heuristic behavior and prevents 100x scale mistakes

### 3.3 Typed Provider and Interpolation Configuration

Decision:
- provider_type and interpolation become enums in config
- mapper performs strict dispatch and does not silently correct invalid values

Rules:
- ProviderType enum: CSV, DATABASE, API
- interpolation enum aligns with abides-rohan InterpolationStrategy values
- invalid config should fail at validation time (Pydantic) or mapper resolution with explicit error messages

Rationale:
- fail-fast behavior is safer for simulation integrity

### 3.4 Pluggable Provider Protocol

Decision:
- introduce FundamentalDataProvider protocol for all historical providers

Protocol contract:
- get_fundamental_series(symbol, start, end) -> pd.Series
- list_available(...) -> list of provider-specific dataset descriptors

All providers must:
- return normalized pd.Series with DatetimeIndex and integer cents
- call shared normalization utility

## 4. Target Architecture

### 4.1 New/Updated Types

In simulation data models/config:
- PriceUnit enum: CENTS, DOLLARS
- ProviderType enum (moved/used as authoritative): CSV, DATABASE, API
- InterpolationMode enum mapped to abides-rohan strategy values
- HistoricalOracleSettings extended with provider-specific nested settings:
  - csv: file_path, symbol, unit, timezone
  - database: dataset_id, optional symbol override
  - api: provider_name, symbol, start/end, credentials/env-key reference

DTOs:
- FundamentalDataset metadata includes:
  - symbol
  - source provenance
  - row_count
  - start_time, end_time
  - sampling_freq (reintroduced)
  - price_unit
  - timezone policy used

### 4.2 Provider Implementations

CSV provider:
- reads canonical timestamp, price columns
- supports explicit unit and timezone policy
- validates required columns and non-empty data

Database provider:
- resolves dataset by dataset_id from new curated data tables
- loads points ordered by timestamp
- returns normalized series
- supports list_available for UI dataset dropdown

API provider:
- thin adapter boundary only (Alpaca/Polygon style)
- maps provider payload to canonical dataframe (timestamp, price)
- applies normalization and returns series
- for this iteration, concrete adapter can be minimal but the provider class and contract are complete

### 4.3 Database Storage for Historical Datasets

Add SQLAlchemy tables under framework database models (same DB stack):
- fundamental_datasets:
  - dataset_id UUID PK
  - symbol
  - source_name
  - provenance
  - sampling_freq
  - start_time, end_time
  - timezone
  - price_unit
  - created_at
- fundamental_data_points:
  - id PK
  - dataset_id FK
  - timestamp
  - price_cents (int)
  - unique(dataset_id, timestamp)

Repository/service layer:
- create_dataset_from_series(metadata, series)
- fetch_series(dataset_id, start, end)
- list_datasets(symbol optional)

No migration compatibility constraints: table creation can be direct and destructive reset is acceptable.

### 4.4 Mapper Dispatch

AbidesConfigMapper historical path:
- resolve provider by HistoricalOracleSettings.provider_type
- build provider with provider-specific settings
- build ExternalDataOracle with strict interpolation mapping
- no silent interpolation fallback

Failure behavior:
- missing required provider settings -> ValueError with field context
- unsupported provider type -> ValueError with accepted values

## 5. UI Design (Phase 1 scope only)

Sidebar oracle section:
- oracle type toggle remains
- historical mode adds provider type selector
- provider-specific forms:
  - CSV: dataset dropdown or file path
  - DATABASE: dataset dropdown from list_available
  - API: provider selector, symbol, key reference
- preview chart uses resolved provider sample and clearly shows provenance metadata
- apply validation blocks execution if provider config is invalid

Implementation note:
- keep edits localized in existing file for this phase; full UI decomposition is Phase 3

## 6. Incremental Implementation Plan

Order follows interface-first workflow.

Stage A - Interfaces and Config Typing
1. add enums (ProviderType, PriceUnit, InterpolationMode)
2. update HistoricalOracleSettings to typed nested provider settings
3. add normalization options struct (unit, timezone policy)
4. add provider protocol

Stage B - Core Implementations
1. update normalization logic (timezone-safe and unit-aware)
2. update CSV provider to pass normalization options
3. add DB schema for historical datasets and repository helpers
4. add DatabaseDataProvider
5. add ApiDataProvider scaffold and canonical response mapper boundary

Stage C - Mapper Integration
1. refactor _build_historical_oracle to provider dispatch factory
2. remove interpolation fallback and enforce strict mode
3. improve error messages

Stage D - UI Integration
1. add provider selector and provider-specific inputs
2. wire list_available for CSV/DB
3. add apply-time validation and preview behavior updates

Stage E - Documentation
1. add data pipeline reference doc
2. add provider development guide
3. update phase checklist status in plan file

## 7. Verification Strategy

### 7.1 Unit Tests

Normalization tests:
- tz-aware conversion preserves exchange-local wall-clock time
- dollars to cents conversion
- cents passthrough conversion
- unknown unit fails
- recentering works after conversion
- monotonic and positive checks

Provider tests:
- CSV happy path, missing columns, wrong symbol, out-of-range slice
- DB provider fetch and list_available
- API provider mapping and error propagation

Mapper tests:
- oracle type synthetic/historical dispatch
- provider dispatch by ProviderType
- invalid interpolation fails
- missing provider settings fail with explicit error

### 7.2 Integration Tests

- full historical simulation with CSV provider
- full historical simulation with DB provider
- partial/failure case for API provider mocked adapter
- coherence guard: market prices remain within reasonable band

### 7.3 Tooling Checks

- targeted pytest subsets while iterating
- pyright on modified modules
- ruff checks
- full pre-commit before finalization

## 8. Risks and Mitigations

Risk: timezone misconfiguration by users
- mitigation: default explicit exchange timezone and validation errors

Risk: API rate limits / flaky external dependency
- mitigation: adapter mocking in tests and timeout/retry policy boundaries

Risk: DB table growth for high-frequency datasets
- mitigation: indexes on dataset_id+timestamp and bounded query windows

## 9. File-Level Change Plan

Config:
- src/rohan/config/agent_settings.py
- src/rohan/config/__init__.py

Simulation data layer:
- src/rohan/simulation/data/models.py
- src/rohan/simulation/data/normalization.py
- src/rohan/simulation/data/csv_provider.py
- src/rohan/simulation/data/database_provider.py (new)
- src/rohan/simulation/data/api_provider.py (new)
- src/rohan/simulation/data/provider_protocol.py (new)
- src/rohan/simulation/data/__init__.py

Mapper:
- src/rohan/simulation/abides_impl/abides_config_mapper.py

Database:
- src/rohan/framework/database/models.py
- src/rohan/framework/database/init_db.py (if needed)
- src/rohan/framework/database/repositories/fundamental_dataset_repository.py (new, if repository folder exists)

UI:
- src/rohan/ui/0_Terminal.py

Tests:
- tests/test_data_pipeline.py
- tests/test_abides_config_mapper.py
- tests/test_historical_integration.py
- tests/test_database_historical_provider.py (new)
- tests/test_api_historical_provider.py (new)

Docs:
- docs/technical/data_pipeline_reference.md (new)
- docs/technical/agent_plugin_development_guide.md (if touched this phase, else defer)

## 10. Definition of Done

Phase 1 hardening is complete when:
- historical pipeline is typed and fail-fast
- provider dispatch supports CSV/DB/API classes through a common protocol
- normalization is timezone-safe and unit-explicit
- tests cover standard and edge cases for normalization, provider dispatch, and historical simulation
- pre-commit, pyright, and targeted tests pass
