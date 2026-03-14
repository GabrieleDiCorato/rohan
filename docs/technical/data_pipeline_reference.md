# Historical Data Pipeline Reference

## Overview

The historical pipeline feeds ExternalDataOracle through pluggable providers that all return the same normalized representation:
- pandas Series
- DatetimeIndex (timezone-naive, market-clock aligned)
- integer cents

## Normalization Contract

Function: normalize_fundamental_series

Inputs:
- raw_series: unnormalized prices indexed by timestamps
- price_unit: CENTS or DOLLARS
- source_timezone: IANA timezone used when source timestamps are timezone-aware
- r_bar: optional recenter target

Rules:
- timestamps are parsed into DatetimeIndex
- timezone-aware indexes are converted to source_timezone and then made naive
- prices are converted to integer cents using explicit price_unit
- missing values are forward/backward filled
- validation enforces monotonic timestamps and strictly positive prices

## Provider Interface

FundamentalDataProvider protocol:
- get_fundamental_series(symbol, start, end) -> pd.Series
- list_available(source=None) -> list[str]

All providers must call normalization before returning data.

## Implemented Providers

CsvDataProvider
- Input columns:
  - timestamp
  - price_cents when unit=CENTS
  - price when unit=DOLLARS
- list_available scans a local directory for csv files

DatabaseDataProvider
- Reads from:
  - fundamental_data_points(dataset_id, timestamp, price_cents)
  - fundamental_datasets for discovery
- Raises explicit errors when tables are missing or dataset has no points

ApiDataProvider
- Requires an injected fetcher adapter
- Expects canonical payload columns timestamp and price
- Performs normalization with explicit unit/timezone settings

## Oracle Configuration

HistoricalOracleSettings now includes:
- provider_type: CSV, DATABASE, API
- interpolation: ffill, nearest, linear
- provider-specific nested settings:
  - csv
  - database
  - api

AbidesConfigMapper strictly dispatches by provider_type and fails fast on missing required fields.

## UI Behavior

Terminal historical oracle mode supports:
- provider selector
- provider-specific inputs
- interpolation selection
- optional recenter to value-agent r_bar
- csv preview chart

## Testing Coverage

Current tests validate:
- cents and dollars conversion
- timezone-aware timestamp conversion preserving market clock
- historical oracle CSV dispatch and fail-fast missing path
- full historical simulation execution with ExternalDataOracle
