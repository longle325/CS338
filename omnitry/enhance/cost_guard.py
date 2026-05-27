import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


class BudgetExceeded(RuntimeError):
    pass


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class CostConfig:
    budget_usd: float = 100.0
    soft_budget_usd: float = 95.0
    input_usd_per_1m: float = 5.0
    output_usd_per_1m: float = 20.0
    request_overhead_usd: float = 0.0
    image_input_tokens: int = 1700
    adjust_with_actual_usage: bool = True

    @classmethod
    def from_env(cls, prefix="LLM_LABEL_"):
        budget = _env_float(f"{prefix}BUDGET_USD", 100.0)
        soft_default = min(budget, _env_float(f"{prefix}SOFT_BUDGET_USD", budget * 0.95))
        return cls(
            budget_usd=budget,
            soft_budget_usd=soft_default,
            input_usd_per_1m=_env_float(f"{prefix}INPUT_USD_PER_1M", 5.0),
            output_usd_per_1m=_env_float(f"{prefix}OUTPUT_USD_PER_1M", 20.0),
            request_overhead_usd=_env_float(f"{prefix}REQUEST_OVERHEAD_USD", 0.0),
            image_input_tokens=_env_int(f"{prefix}IMAGE_INPUT_TOKENS", 1700),
            adjust_with_actual_usage=os.environ.get(f"{prefix}ADJUST_WITH_ACTUAL_USAGE", "1").lower()
            not in {"0", "false", "no"},
        )

    def estimate(self, input_tokens: int, max_output_tokens: int, image_count: int = 0) -> float:
        total_input_tokens = int(input_tokens) + int(image_count) * self.image_input_tokens
        input_cost = total_input_tokens * self.input_usd_per_1m / 1_000_000
        output_cost = int(max_output_tokens) * self.output_usd_per_1m / 1_000_000
        return float(self.request_overhead_usd + input_cost + output_cost)

    def usage_cost(self, usage: Optional[Dict]) -> Optional[float]:
        if not usage:
            return None
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            total = usage.get("total_tokens")
            if total is None:
                return None
            input_tokens = total
            output_tokens = 0
        return float(
            self.request_overhead_usd
            + int(input_tokens) * self.input_usd_per_1m / 1_000_000
            + int(output_tokens) * self.output_usd_per_1m / 1_000_000
        )


def estimate_text_tokens(text: str) -> int:
    # Conservative tokenizer-free estimate. English JSON/prompt text is usually 3-5 chars/token.
    return max(1, int(math.ceil(len(text or "") / 3.5)))


class CostLedger:
    def __init__(self, state_path, events_path=None, config: Optional[CostConfig] = None):
        self.state_path = Path(state_path)
        self.events_path = Path(events_path) if events_path else self.state_path.with_suffix(".jsonl")
        self.config = config or CostConfig.from_env()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self):
        if self.state_path.is_file():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {}
        state.setdefault("spent_usd", 0.0)
        state.setdefault("reserved", {})
        state.setdefault("requests_reserved", 0)
        state.setdefault("requests_completed", 0)
        state.setdefault("requests_blocked", 0)
        state.setdefault("budget_usd", self.config.budget_usd)
        state.setdefault("soft_budget_usd", self.config.soft_budget_usd)
        return state

    def _save_state(self):
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(self.state_path)

    def _event(self, event_type, **payload):
        payload = {"time": time.time(), "event": event_type, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @property
    def spent_usd(self):
        return float(self.state.get("spent_usd", 0.0))

    def remaining_to_soft_budget(self):
        return max(0.0, self.config.soft_budget_usd - self.spent_usd)

    def reserve(self, request_id: str, estimated_cost_usd: float, metadata: Optional[Dict] = None):
        estimated_cost_usd = float(estimated_cost_usd)
        projected = self.spent_usd + estimated_cost_usd
        limit = min(self.config.soft_budget_usd, self.config.budget_usd)
        if projected >= limit:
            self.state["requests_blocked"] = int(self.state.get("requests_blocked", 0)) + 1
            self._save_state()
            self._event(
                "blocked",
                request_id=request_id,
                estimated_cost_usd=estimated_cost_usd,
                spent_usd=self.spent_usd,
                projected_spent_usd=projected,
                soft_budget_usd=self.config.soft_budget_usd,
                budget_usd=self.config.budget_usd,
                metadata=metadata or {},
            )
            raise BudgetExceeded(
                f"Budget guard blocked request {request_id}: projected ${projected:.4f} "
                f"would reach limit ${limit:.4f}."
            )

        self.state["spent_usd"] = projected
        self.state["requests_reserved"] = int(self.state.get("requests_reserved", 0)) + 1
        self.state.setdefault("reserved", {})[request_id] = estimated_cost_usd
        self._save_state()
        self._event(
            "reserved",
            request_id=request_id,
            estimated_cost_usd=estimated_cost_usd,
            spent_usd=self.spent_usd,
            metadata=metadata or {},
        )

    def complete(self, request_id: str, usage: Optional[Dict] = None, metadata: Optional[Dict] = None):
        estimated = float(self.state.get("reserved", {}).pop(request_id, 0.0))
        actual = self.config.usage_cost(usage)
        if actual is not None and self.config.adjust_with_actual_usage:
            self.state["spent_usd"] = max(0.0, self.spent_usd - estimated + actual)
        self.state["requests_completed"] = int(self.state.get("requests_completed", 0)) + 1
        self._save_state()
        self._event(
            "completed",
            request_id=request_id,
            estimated_cost_usd=estimated,
            actual_cost_usd=actual,
            usage=usage or {},
            spent_usd=self.spent_usd,
            metadata=metadata or {},
        )

    def fail(self, request_id: str, error: str, metadata: Optional[Dict] = None):
        self._event(
            "failed",
            request_id=request_id,
            error=error,
            spent_usd=self.spent_usd,
            metadata=metadata or {},
        )
