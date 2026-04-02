"""Tests for StrategySpec serialisation, compilation, and round-tripping.

Covers:
- Construction and immutability (frozen Pydantic model)
- Compilation via the sandboxed exec pipeline
- JSON round-trip (model_dump / model_validate)
- Pickle round-trip (serialisability for run_batch)
- Hash stability
- Error handling (bad code, missing class, forbidden imports)
"""

from __future__ import annotations

import pickle

import pytest
from pydantic import ValidationError

from rohan.exceptions import StrategyExecutionError, StrategyValidationError
from rohan.simulation.models.strategy_spec import StrategySpec

# ---------------------------------------------------------------------------
# Valid strategy source code used across tests
# ---------------------------------------------------------------------------

SIMPLE_STRATEGY_CODE = '''
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, Order, OrderAction, StrategicAgent,
)

class SimpleTestStrategy(StrategicAgent):
    """Minimal strategy that does nothing — just satisfies the protocol."""

    def initialize(self, config: AgentConfig) -> None:
        self.config = config

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
'''


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_create_with_defaults(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        assert spec.source_code == SIMPLE_STRATEGY_CODE
        assert spec.class_name == "SimpleTestStrategy"
        assert spec.params == {}

    def test_create_with_params(self):
        spec = StrategySpec(
            source_code=SIMPLE_STRATEGY_CODE,
            class_name="SimpleTestStrategy",
            params={"threshold": 0.5},
        )
        assert spec.params == {"threshold": 0.5}

    def test_frozen_model(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        with pytest.raises(ValidationError):
            spec.source_code = "other code"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class TestCompilation:
    def test_compile_returns_class(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        cls = spec.compile()
        assert isinstance(cls, type)
        assert cls.__name__ == "SimpleTestStrategy"

    def test_compiled_class_is_instantiable(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        cls = spec.compile()
        instance = cls()
        # Should have all protocol methods
        assert hasattr(instance, "initialize")
        assert hasattr(instance, "on_tick")
        assert hasattr(instance, "on_market_data")
        assert hasattr(instance, "on_order_update")
        assert hasattr(instance, "on_simulation_end")

    def test_compile_bad_code_raises(self):
        spec = StrategySpec(source_code="def foo(: invalid syntax", class_name="Foo")
        with pytest.raises((StrategyValidationError, StrategyExecutionError, SyntaxError)):
            spec.compile()

    def test_compile_missing_class_raises(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="NonExistentClass")
        with pytest.raises(StrategyValidationError, match="not found"):
            spec.compile()

    def test_compile_forbidden_import_raises(self):
        bad_code = "import os\nclass Bad:\n    pass"
        spec = StrategySpec(source_code=bad_code, class_name="Bad")
        with pytest.raises(StrategyValidationError):
            spec.compile()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_json_round_trip(self):
        spec = StrategySpec(
            source_code=SIMPLE_STRATEGY_CODE,
            class_name="SimpleTestStrategy",
            params={"threshold": 0.5},
        )
        json_str = spec.model_dump_json()
        restored = StrategySpec.model_validate_json(json_str)
        assert restored == spec

    def test_dict_round_trip(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        d = spec.model_dump()
        restored = StrategySpec.model_validate(d)
        assert restored == spec

    def test_pickle_round_trip(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        pickled = pickle.dumps(spec)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored == spec

    def test_hash_stability(self):
        spec1 = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        spec2 = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        assert hash(spec1) == hash(spec2)

    def test_different_code_different_hash(self):
        spec1 = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        spec2 = StrategySpec(source_code="class X:\n  pass", class_name="X")
        assert hash(spec1) != hash(spec2)

    def test_usable_as_dict_key(self):
        spec = StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")
        d = {spec: "value"}
        assert d[spec] == "value"
