"""
Sub-Problem E — Unit Tests
Covers: feature engineering, drift detector, model registry, API validation.
Target: ≥70% code coverage.
Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pytest

from src.api import DriftDetector, ModelRegistry, TransactionInput, _risk_tier


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_labels(rng):
    y = rng.integers(0, 2, 200)
    y[:10] = 1
    return y


@pytest.fixture
def feature_matrix(rng):
    return rng.standard_normal((200, 10)).astype(np.float32)


@pytest.fixture
def dummy_model(feature_matrix, binary_labels):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(max_iter=200, random_state=0)
    m.fit(feature_matrix, binary_labels)
    return m


@pytest.fixture
def tmp_registry(tmp_path):
    return ModelRegistry(registry_dir=tmp_path)


class TestEvaluationMetrics:
    def test_expected_cost_all_correct(self):
        from src.evaluation import expected_cost

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        cost = expected_cost(y_true, y_prob, threshold=0.5)
        assert cost == 0

    def test_expected_cost_all_fn(self):
        from src.evaluation import expected_cost

        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1])
        cost = expected_cost(y_true, y_prob, threshold=0.5)
        assert cost == 3 * 200

    def test_metrics_at_threshold_returns_keys(self):
        from src.evaluation import metrics_at_threshold

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = metrics_at_threshold(y_true, y_prob, 0.5)
        for key in ["threshold", "precision", "recall", "f1", "expected_cost"]:
            assert key in result

    def test_full_threshold_report_returns_dataframe(self):
        from src.evaluation import full_threshold_report
        import pandas as pd

        y_true = np.array([0] * 90 + [1] * 10)
        y_prob = np.clip(
            np.random.default_rng(0).normal(loc=np.where(y_true == 1, 0.7, 0.3), scale=0.1),
            0,
            1,
        )
        df, bt_f1, bt_cost = full_threshold_report(y_true, y_prob)
        assert isinstance(df, pd.DataFrame)
        assert 0 < bt_f1 < 1
        assert 0 < bt_cost < 1

    def test_threshold_bounds(self):
        from src.evaluation import full_threshold_report

        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.linspace(0, 1, 100)
        df, bt_f1, bt_cost = full_threshold_report(y_true, y_prob)
        assert df["threshold"].between(0, 1).all()


class TestDriftDetector:
    def test_fit_stores_stats(self, feature_matrix):
        dd = DriftDetector()
        names = [f"f{i}" for i in range(feature_matrix.shape[1])]
        dd.fit(feature_matrix, names)
        assert len(dd.reference_stats) == 10
        assert "mean" in dd.reference_stats["f0"]

    def test_no_drift_same_distribution(self, feature_matrix):
        names = [f"f{i}" for i in range(feature_matrix.shape[1])]
        dd = DriftDetector()
        dd.fit(feature_matrix, names)
        result = dd.check(feature_matrix, names)
        assert result["status"] == "ok"
        assert result["n_drifted"] == 0

    def test_drift_detected_on_shifted_data(self, feature_matrix):
        names = [f"f{i}" for i in range(feature_matrix.shape[1])]
        dd = DriftDetector()
        dd.fit(feature_matrix, names)
        shifted = feature_matrix + 10.0
        result = dd.check(shifted, names)
        assert result["status"] == "drift_detected"
        assert result["n_drifted"] > 0

    def test_no_reference_returns_status(self, feature_matrix):
        dd = DriftDetector()
        names = [f"f{i}" for i in range(feature_matrix.shape[1])]
        result = dd.check(feature_matrix, names)
        assert result["status"] == "no_reference"

    def test_psi_zero_on_identical(self, feature_matrix):
        dd = DriftDetector()
        psi = dd._psi(feature_matrix[:, 0], feature_matrix[:, 0])
        assert psi < 0.01


class TestModelRegistry:
    def test_register_and_load(self, dummy_model, tmp_registry, feature_matrix):
        version = tmp_registry.register(dummy_model, {"algo": "LR", "auc": 0.82})
        loaded = tmp_registry.load(version)
        preds = loaded.predict_proba(feature_matrix[:5])
        assert preds.shape == (5, 2)

    def test_current_version_updated(self, dummy_model, tmp_registry):
        tmp_registry.register(dummy_model, {}, version="v_test_1")
        tmp_registry.register(dummy_model, {}, version="v_test_2")
        assert tmp_registry.manifest["current"] == "v_test_2"
        assert tmp_registry.manifest["previous"] == "v_test_1"

    def test_rollback_restores_previous(self, dummy_model, tmp_registry):
        tmp_registry.register(dummy_model, {}, version="v_A")
        tmp_registry.register(dummy_model, {}, version="v_B")
        rolled = tmp_registry.rollback()
        assert rolled == "v_A"
        assert tmp_registry.manifest["current"] == "v_A"

    def test_rollback_no_previous_raises(self, dummy_model, tmp_registry):
        tmp_registry.register(dummy_model, {}, version="v_only")
        tmp_registry.manifest["previous"] = None
        tmp_registry._save_manifest()
        with pytest.raises(ValueError, match="No previous version"):
            tmp_registry.rollback()

    def test_load_nonexistent_version_raises(self, tmp_registry):
        with pytest.raises(ValueError, match="not found"):
            tmp_registry.load("v_ghost")

    def test_list_versions(self, dummy_model, tmp_registry):
        tmp_registry.register(dummy_model, {}, version="v1")
        tmp_registry.register(dummy_model, {}, version="v2")
        versions = tmp_registry.list_versions()
        assert len(versions) >= 2


class TestInputValidation:
    def test_valid_transaction(self):
        t = TransactionInput(transaction_id="tx_001", features=[1.0, 2.5, -0.3])
        assert len(t.features) == 3

    def test_nan_in_features_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransactionInput(features=[1.0, float("nan"), 3.0])

    def test_inf_in_features_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransactionInput(features=[1.0, float("inf")])

    def test_empty_features_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransactionInput(features=[])

    def test_risk_tier_classification(self):
        assert _risk_tier(0.9) == "HIGH"
        assert _risk_tier(0.5) == "MEDIUM"
        assert _risk_tier(0.1) == "LOW"
        assert _risk_tier(0.8) == "HIGH"
        assert _risk_tier(0.4) == "MEDIUM"
        assert _risk_tier(0.39) == "LOW"


class TestFeatureEngineering:
    def test_engineer_features_adds_columns(self):
        from src.features import engineer_features
        import pandas as pd

        df = pd.DataFrame(
            {
                "Time": [0.0],
                "Amount": [100.0],
                **{f"V{i}": [0.1] for i in range(1, 29)},
                "Class": [0],
            }
        )
        out = engineer_features(df)
        assert "fe_log_amount" in out.columns
        assert out["fe_log_amount"].notna().all()

    def test_no_nan_after_scaling(self, feature_matrix):
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        X = sc.fit_transform(feature_matrix)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_feature_matrix_shape_preserved(self, feature_matrix):
        from sklearn.preprocessing import RobustScaler

        sc = RobustScaler()
        X = sc.fit_transform(feature_matrix)
        assert X.shape == feature_matrix.shape

    def test_class_imbalance_smote(self):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
        X = np.random.randn(200, 5)
        y = np.array([0] * 180 + [1] * 20)
        sm = SMOTE(random_state=0, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        assert (y_res == 0).sum() == (y_res == 1).sum()

    def test_prediction_probability_bounds(self, dummy_model, feature_matrix):
        probs = dummy_model.predict_proba(feature_matrix)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()


class TestInferencePipeline:
    def test_raw_features_pipeline(self, dummy_model, feature_matrix):
        from sklearn.preprocessing import StandardScaler
        from src.inference import FraudDetectionPipeline

        scaler = StandardScaler()
        scaler.fit(feature_matrix)
        pipeline = FraudDetectionPipeline(dummy_model, scaler, [f"f{i}" for i in range(10)])
        probs = pipeline.predict_proba(feature_matrix[:3])
        assert probs.shape == (3, 2)


class TestApiEndpoints:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "uptime_seconds" in body

    def test_metrics_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "fraud_api_requests_total" in resp.text

    def test_predict_without_model_returns_503(self):
        from fastapi.testclient import TestClient
        import src.api as api_module

        api_module._model = None
        client = TestClient(api_module.app)
        resp = client.post("/predict", json={"features": [0.0] * 30})
        assert resp.status_code == 503

    def test_monitoring_summary_empty(self):
        from fastapi.testclient import TestClient
        from src.api import app, _monitor

        _monitor.predictions.clear()
        _monitor.latencies.clear()
        client = TestClient(app)
        resp = client.get("/monitoring/summary")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"


class TestMlops:
    def test_retrain_trigger_on_drift(self):
        from src.mlops import RetrainTrigger

        trigger = RetrainTrigger()
        result = trigger.should_retrain(n_drifted_features=3)
        assert result["should_retrain"] is True
        assert "drift_detected_3_features" in result["reasons"]

    def test_retrain_trigger_passes_when_healthy(self):
        from src.mlops import RetrainTrigger

        trigger = RetrainTrigger()
        result = trigger.should_retrain(current_auroc=0.92, current_f1=0.75, n_drifted_features=0)
        assert result["should_retrain"] is False

    def test_model_promotion_gate(self):
        from src.mlops import RetrainTrigger

        trigger = RetrainTrigger()
        ok = trigger.promote_model(new_auroc=0.91, old_auroc=0.90, new_pr_auc=0.65)
        assert ok["promote"] is True

    def test_monitoring_dashboard_summary(self):
        from src.mlops import MonitoringDashboard

        dash = MonitoringDashboard()
        dash.record(0.8, True, 12.5)
        dash.record(0.2, False, 8.0)
        summary = dash.summary()
        assert summary["total_predictions"] == 2
        assert summary["fraud_rate"] == 0.5


class TestModelLoader:
    def test_bootstrap_registry_from_notebook_models(self, tmp_path):
        from pathlib import Path
        import shutil
        from src.api import ModelRegistry
        from src.model_loader import bootstrap_registry

        src_models = Path("notebooks/saved_models")
        if not (src_models / "xgboost.joblib").exists():
            pytest.skip("Notebook models not trained yet")

        models_dir = tmp_path / "models"
        shutil.copytree(src_models, models_dir)
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        version = bootstrap_registry(registry, models_dir=models_dir)
        assert version is not None
        pipeline = registry.load(version)
        raw = np.zeros((1, 30), dtype=np.float32)
        probs = pipeline.predict_proba(raw)
        assert probs.shape == (1, 2)

    def test_predict_with_loaded_pipeline(self, tmp_path):
        from pathlib import Path
        import shutil
        from fastapi.testclient import TestClient
        import src.api as api_module
        from src.api import ModelRegistry
        from src.model_loader import bootstrap_registry

        src_models = Path("notebooks/saved_models")
        if not (src_models / "xgboost.joblib").exists():
            pytest.skip("Notebook models not trained yet")

        models_dir = tmp_path / "models"
        shutil.copytree(src_models, models_dir)
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        version = bootstrap_registry(registry, models_dir=models_dir)
        api_module._registry = registry
        api_module._model = registry.load(version)
        api_module._model_ver = version

        client = TestClient(api_module.app)
        resp = client.post("/predict", json={"transaction_id": "t1", "features": [0.0] * 30})
        assert resp.status_code == 200
        body = resp.json()
        assert "fraud_probability" in body
        assert body["transaction_id"] == "t1"


class TestEndToEndPipeline:
    def test_train_predict_pipeline(self, feature_matrix, binary_labels):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(
            feature_matrix, binary_labels, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        assert len(probs) == len(y_te)
        assert probs.min() >= 0 and probs.max() <= 1

    def test_registry_persist_and_reload(self, dummy_model, tmp_path, feature_matrix):
        reg = ModelRegistry(tmp_path)
        v = reg.register(dummy_model, {"auc": 0.9})
        reg2 = ModelRegistry(tmp_path)
        m2 = reg2.load(v)
        p1 = dummy_model.predict_proba(feature_matrix[:3])
        p2 = m2.predict_proba(feature_matrix[:3])
        np.testing.assert_array_almost_equal(p1, p2)

    def test_drift_full_workflow(self, feature_matrix):
        names = [f"f{i}" for i in range(10)]
        dd = DriftDetector()
        dd.fit(feature_matrix, names)
        r1 = dd.check(feature_matrix + 0.01, names)
        r2 = dd.check(feature_matrix * 100, names)
        assert r2["n_drifted"] >= r1["n_drifted"]
        assert r2["n_drifted"] > 0
