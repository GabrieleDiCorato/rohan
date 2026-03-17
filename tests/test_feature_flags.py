from rohan.config.feature_flags import FeatureFlags, FeatureFlagSnapshot, feature_flags_dict, feature_flags_snapshot, get_feature_flags


def test_feature_flags_defaults():
    FeatureFlags.model_rebuild(force=True)
    get_feature_flags.cache_clear()
    flags = get_feature_flags()
    assert flags.llm_explainer_tiers_v1 is True
    assert flags.explicit_terminal_reasons_v1 is True
    assert flags.baseline_cache_v1 is True
    assert flags.llm_telemetry_v1 is True


def test_feature_flags_dict_shape():
    get_feature_flags.cache_clear()
    flags = feature_flags_dict()
    assert set(flags.keys()) == {
        "llm_explainer_tiers_v1",
        "explicit_terminal_reasons_v1",
        "baseline_cache_v1",
        "llm_telemetry_v1",
    }


def test_feature_flags_snapshot_is_typed_model():
    get_feature_flags.cache_clear()
    snapshot = feature_flags_snapshot()
    assert isinstance(snapshot, FeatureFlagSnapshot)
    assert snapshot.explicit_terminal_reasons_v1 is True
