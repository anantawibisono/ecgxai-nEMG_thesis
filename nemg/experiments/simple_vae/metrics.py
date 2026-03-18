from __future__ import annotations

from torchmetrics import MeanMetric, MetricCollection


def build_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "loss": MeanMetric(),
            "recon": MeanMetric(),
            "kl": MeanMetric(),
        }
    )