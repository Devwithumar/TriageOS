"""Small dependency-free Prometheus exposition for service-level metrics."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Mapping


_DEFAULT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


class MetricsRegistry:
    """Thread-safe process metrics with deliberately bounded label usage."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, object]] = {}

    def increment(self, name: str, labels: Mapping[str, str] | None = None, value: float = 1) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        key = _key(name, labels)
        with self._lock:
            state = self._histograms.setdefault(
                key,
                {"count": 0.0, "sum": 0.0, "buckets": [0.0 for _ in _DEFAULT_BUCKETS]},
            )
            state["count"] = float(state["count"]) + 1
            state["sum"] = float(state["sum"]) + value
            bucket_counts = state["buckets"]
            assert isinstance(bucket_counts, list)
            for index, boundary in enumerate(_DEFAULT_BUCKETS):
                if value <= boundary:
                    bucket_counts[index] += 1

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            histograms = {
                key: {
                    "count": float(value["count"]),
                    "sum": float(value["sum"]),
                    "buckets": list(value["buckets"]),
                }
                for key, value in self._histograms.items()
            }

        lines: list[str] = []
        emitted_types: set[str] = set()
        for (name, labels), value in sorted(counters.items()):
            if name not in emitted_types:
                lines.append(f"# TYPE {name} counter")
                emitted_types.add(name)
            lines.append(f"{name}{_render_labels(labels)} {value:g}")

        for (name, labels), state in sorted(histograms.items()):
            if name not in emitted_types:
                lines.append(f"# TYPE {name} histogram")
                emitted_types.add(name)
            for boundary, count in zip(_DEFAULT_BUCKETS, state["buckets"]):
                bucket_labels = _key_labels({**dict(labels), "le": _format_number(boundary)})
                lines.append(f"{name}_bucket{_render_labels(bucket_labels)} {count:g}")
            inf_labels = _key_labels({**dict(labels), "le": "+Inf"})
            lines.append(f"{name}_bucket{_render_labels(inf_labels)} {state['count']:g}")
            lines.append(f"{name}_sum{_render_labels(labels)} {state['sum']:g}")
            lines.append(f"{name}_count{_render_labels(labels)} {state['count']:g}")
        return "\n".join(lines) + ("\n" if lines else "")


def _key(name: str, labels: Mapping[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ValueError(f"invalid metric name: {name!r}")
    return name, _key_labels(labels or {})


def _key_labels(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape(value)}"' for key, value in labels)
    return "{" + rendered + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: float) -> str:
    return f"{value:g}"
