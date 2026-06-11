"""
Sub-Problem E — MLOps: Retrain Trigger, Structured Logging, Monitoring
Implements:
- Automatic retrain trigger based on drift / performance degradation
- Structured JSON logging throughout
- Lightweight monitoring dashboard data endpoint
- Retrain strategy documentation
"""

import json, logging, time, os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np

# ── Structured Logger ─────────────────────────

class StructuredLogger:
    """JSON-structured logger wrapping Python's standard logging."""

    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(message)s')  # raw JSON

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def _emit(self, level: str, event: str, **kwargs):
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            "service": "fraud-detection",
            **kwargs,
        }
        getattr(self.logger, level.lower())(json.dumps(record))

    def info(self, event, **kwargs):   self._emit("INFO",    event, **kwargs)
    def warning(self, event, **kwargs):self._emit("WARNING", event, **kwargs)
    def error(self, event, **kwargs):  self._emit("ERROR",   event, **kwargs)
    def debug(self, event, **kwargs):  self._emit("DEBUG",   event, **kwargs)


logger = StructuredLogger("mlops", log_file="logs/mlops.log" if Path("logs").exists() else None)


# ── Retrain Trigger ───────────────────────────

class RetrainTrigger:
    """
    Monitors model performance and data drift; raises retrain signal
    when conditions breach configurable thresholds.

    Retrain Strategy (documented):
    ────────────────────────────────────────────────────────────────
    TRIGGER CONDITIONS (any one sufficient):
      1. PSI > 0.20 on ≥ 2 features over a rolling 7-day window
      2. AUROC on labelled feedback data drops > 5 pp below baseline
      3. F1 score drops > 8 pp below baseline
      4. Fraud rate in production deviates > 3σ from training fraud rate
      5. Scheduled: monthly full retrain regardless of drift

    RETRAIN PIPELINE:
      1. Pull last N days of labelled transactions (N = 90 default)
      2. Re-run feature engineering (scaling, SMOTE)
      3. Hyperparameter search on validation fold (Optuna, 50 trials)
      4. Shadow-deploy new model: run both old + new, compare on live traffic
      5. Promote new model if: new AUROC ≥ old AUROC – 0.01 AND PR-AUC ≥ 0.60
      6. Register promoted model in MLflow; archive old version
      7. Send alert (Slack/PagerDuty) with performance comparison report

    ROLLBACK:
      - Automatic if p99 latency > 150 ms post-deploy
      - Automatic if new model AUROC < 0.75 on first 1000 scored transactions
      - Manual via POST /model/rollback endpoint
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self,
                 baseline_auroc: float = 0.90,
                 baseline_f1: float = 0.72,
                 auroc_drop_threshold: float = 0.05,
                 f1_drop_threshold: float = 0.08,
                 drift_feature_threshold: int = 2,
                 check_interval_hours: int = 24):

        self.baseline_auroc = baseline_auroc
        self.baseline_f1 = baseline_f1
        self.auroc_drop_threshold = auroc_drop_threshold
        self.f1_drop_threshold = f1_drop_threshold
        self.drift_threshold = drift_feature_threshold
        self.check_interval = timedelta(hours=check_interval_hours)
        self.last_check: Optional[datetime] = None
        self.retrain_history: list = []

    def should_retrain(self,
                       current_auroc: Optional[float] = None,
                       current_f1: Optional[float] = None,
                       n_drifted_features: int = 0,
                       force: bool = False) -> dict:
        """
        Evaluate retrain conditions and return decision + reasons.
        """
        reasons = []
        now = datetime.utcnow()

        # Scheduled monthly retrain
        if self.last_check and (now - self.last_check) > timedelta(days=30):
            reasons.append("scheduled_monthly_retrain")

        # Performance degradation
        if current_auroc is not None:
            drop = self.baseline_auroc - current_auroc
            if drop > self.auroc_drop_threshold:
                reasons.append(f"auroc_drop_{drop:.3f}_exceeds_{self.auroc_drop_threshold}")

        if current_f1 is not None:
            drop = self.baseline_f1 - current_f1
            if drop > self.f1_drop_threshold:
                reasons.append(f"f1_drop_{drop:.3f}_exceeds_{self.f1_drop_threshold}")

        # Drift
        if n_drifted_features >= self.drift_threshold:
            reasons.append(f"drift_detected_{n_drifted_features}_features")

        # Force flag
        if force:
            reasons.append("manual_force")

        decision = bool(reasons)
        result = {
            "should_retrain": decision,
            "reasons": reasons,
            "evaluated_at": now.isoformat(),
            "current_auroc": current_auroc,
            "current_f1": current_f1,
            "n_drifted_features": n_drifted_features,
        }

        if decision:
            logger.warning("retrain_triggered",
                           reasons=reasons,
                           current_auroc=current_auroc,
                           current_f1=current_f1,
                           n_drifted=n_drifted_features)
            self.retrain_history.append(result)
        else:
            logger.info("retrain_check_passed",
                        current_auroc=current_auroc,
                        current_f1=current_f1)

        self.last_check = now
        return result

    def promote_model(self, new_auroc: float, old_auroc: float,
                      new_pr_auc: float, min_pr_auc: float = 0.60) -> dict:
        """Shadow-deployment promotion gate."""
        passes_auroc = new_auroc >= old_auroc - 0.01
        passes_pr    = new_pr_auc >= min_pr_auc
        promote = passes_auroc and passes_pr

        result = {
            "promote": promote,
            "new_auroc": new_auroc,
            "old_auroc": old_auroc,
            "new_pr_auc": new_pr_auc,
            "passes_auroc_gate": passes_auroc,
            "passes_pr_gate": passes_pr,
        }
        if promote:
            logger.info("model_promoted", **result)
        else:
            logger.warning("model_promotion_rejected", **result)
        return result


# ── Monitoring Stats Aggregator ───────────────

class MonitoringDashboard:
    """Aggregates in-memory stats for lightweight monitoring endpoint."""

    def __init__(self):
        self.predictions: list = []
        self.latencies: list = []
        self.drift_events: list = []

    def record(self, prob: float, is_fraud: bool, latency_ms: float):
        self.predictions.append({"prob": prob, "is_fraud": is_fraud,
                                  "ts": datetime.utcnow().isoformat()})
        self.latencies.append(latency_ms)
        # Keep rolling 10k window
        if len(self.predictions) > 10_000:
            self.predictions = self.predictions[-10_000:]
            self.latencies   = self.latencies[-10_000:]

    def record_drift(self, feature: str, psi: float):
        self.drift_events.append({"feature": feature, "psi": psi,
                                   "ts": datetime.utcnow().isoformat()})

    def summary(self) -> dict:
        if not self.predictions:
            return {"status": "no_data"}
        probs = [p["prob"] for p in self.predictions]
        fraud_flags = [p["is_fraud"] for p in self.predictions]
        lat = self.latencies or [0]
        return {
            "total_predictions": len(self.predictions),
            "fraud_rate": round(sum(fraud_flags) / len(fraud_flags), 4),
            "avg_fraud_prob": round(float(np.mean(probs)), 4),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
            "drift_events_24h": len([e for e in self.drift_events
                                     if e["ts"] > (datetime.utcnow()
                                                   - timedelta(hours=24)).isoformat()]),
            "generated_at": datetime.utcnow().isoformat(),
        }


# ── Retrain strategy markdown doc ────────────
RETRAIN_STRATEGY_DOC = """
# Fraud Detection Model — Retrain Strategy

## Overview
The system continuously monitors production traffic for data drift and performance
degradation, automatically triggering retraining when thresholds are breached.

## Trigger Conditions
| Trigger                  | Threshold         | Check Frequency |
|--------------------------|-------------------|-----------------|
| Data drift (PSI)         | > 0.20 on ≥2 feats | Daily           |
| AUROC degradation        | > 5 pp drop       | Daily           |
| F1 degradation           | > 8 pp drop       | Daily           |
| Scheduled retrain        | Monthly           | Monthly         |
| Manual override          | On-demand         | Any time        |

## Retrain Pipeline
1. **Data Collection** — Pull 90 days labelled transactions; deduplicate; validate schema.
2. **Feature Engineering** — Re-fit scalers; apply SMOTE to training split only.
3. **Training** — Optuna hyperparameter search (50 trials, 5-fold CV, AUC objective).
4. **Shadow Deploy** — New model scores live traffic alongside old; no routing change.
5. **Promotion Gate** — Promote if: new AUC ≥ old AUC − 0.01 AND PR-AUC ≥ 0.60.
6. **Registry** — Register promoted model in MLflow; tag with git SHA + date.
7. **Notification** — Slack/PagerDuty alert with before/after performance report.

## Rollback Conditions
- Automatic: p99 latency > 150 ms post-deploy
- Automatic: new model AUROC < 0.75 on first 1000 live predictions
- Manual: `POST /model/rollback`

## Version Policy
- Keep last 5 model versions in registry
- Archive older versions to cold storage (S3 Glacier / Azure Archive)
- Never delete the "champion" production model until successor promoted
"""

if __name__ == "__main__":
    # Demo
    trigger = RetrainTrigger(baseline_auroc=0.92, baseline_f1=0.75)
    r = trigger.should_retrain(current_auroc=0.85, current_f1=0.74, n_drifted_features=3)
    print(json.dumps(r, indent=2))
    print(RETRAIN_STRATEGY_DOC)