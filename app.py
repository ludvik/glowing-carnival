from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from html import escape as html_escape
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.comparison import (
    compare_two_models,
    confusion_matrix_dataframe,
    discover_runs,
    executive_comparison_table,
    load_json,
    operational_summary_table,
    prediction_distribution_table,
    resultset_paths,
    scored_case_table,
    scored_label_distribution,
    side_by_side_per_class_table,
    token_cost_trace,
    unscored_case_table,
    validate_same_issue_set,
)
from eval_harness.dataset import load_dataset
from eval_harness.prompt import LABELS
from eval_harness.scoring import certified_issues, load_resultset, unscored_analysis, uncertified_issues


st.set_page_config(page_title="Issue Classification Model Evaluator", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(0, 115, 234, 0.08), transparent 32rem),
        linear-gradient(180deg, #f6f7fb 0%, #eef2f7 100%);
}
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d9e0ea;
}
div[data-testid="stMainBlockContainer"] {
    padding-top: 30px;
    max-width: 1480px;
}
h1 {
    color: #182033;
    font-weight: 750;
    letter-spacing: 0;
}
h2, h3 {
    color: #232b3d;
    font-weight: 700;
}
p, li, label, div {
    letter-spacing: 0;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d9e0ea;
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}
div[data-testid="stMetricLabel"] p {
    color: #667085;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #182033;
    font-weight: 750;
}
div[data-testid="stTabs"] > div[role="tablist"] {
    gap: 10px;
    border-bottom: 1px solid #d7dee9;
    padding: 8px 0 0;
    margin-top: 14px;
}
div[data-testid="stTabs"] button[role="tab"] {
    min-height: 50px;
    padding: 0 20px;
    border: 1px solid #d7dee9;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 15px;
    font-weight: 700;
    color: #465268;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #eaf4ff;
    border-color: #0073ea;
    box-shadow: inset 0 -4px 0 #0073ea;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
    color: #0055b8;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #d9e0ea;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
div[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid rgba(0, 115, 234, 0.18);
}
button[kind="primary"] {
    background: #0073ea;
    border-radius: 8px;
    font-weight: 700;
}
div[data-baseweb="select"] > div,
input,
textarea {
    border-radius: 8px !important;
}
.issue-body-box {
    background: #ffffff;
    border: 1px solid #d9e0ea;
    border-radius: 8px;
    color: #1f2937;
    font-size: 14px;
    line-height: 1.55;
    padding: 14px 16px;
    white-space: pre-wrap;
    max-height: 300px;
    overflow: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_json(path: str) -> dict[str, Any]:
    return load_json(Path(path))


@st.cache_data(show_spinner=False)
def cached_dataset(path: str) -> dict[str, Any]:
    return load_dataset(Path(path))


@st.cache_data(show_spinner=False)
def cached_resultset(path: str) -> dict[str, Any]:
    return load_resultset(Path(path))


@st.cache_data(show_spinner=False)
def cached_raw_issue_index(path: str) -> dict[int, dict[str, Any]]:
    raw_path = Path(path)
    if not raw_path.exists():
        return {}
    payload = load_json(raw_path)
    return {int(issue["number"]): issue for issue in payload.get("issues", [])}


def format_number(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:,.1f}"
    return str(value)


def dataframe(df: pd.DataFrame, **kwargs: Any) -> None:
    st.dataframe(arrow_safe_dataframe(df), width="stretch", hide_index=True, **kwargs)


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns before Streamlit serializes via Arrow."""

    if df.empty:
        return df
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].map(display_cell_value)
    return out


def display_cell_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def main() -> None:
    st.title("Issue Classification Model Evaluator")
    st.caption("Compare model quality, cost, latency, and failure behavior on the doctl issue corpus.")

    run_files = discover_runs(ROOT / "runs")
    run_labels = [path.parent.name for path in run_files]
    run_choice = st.sidebar.selectbox("Run", ["New comparison", *run_labels])
    if run_choice == "New comparison":
        render_new_comparison()
        return

    selected_run = run_choice
    run_dir = ROOT / "runs" / selected_run
    run_payload = load_json(run_dir / "run.json")
    resultsets_by_model = effective_resultset_index(run_dir)
    if len(resultsets_by_model) < 2:
        if run_is_active(run_dir, run_payload):
            st.session_state["active_eval_run_id"] = selected_run
            render_run_progress(run_dir)
            return
        st.warning("Selected run has fewer than two model resultsets.")
        return

    models = sorted(resultsets_by_model)
    model_a = st.sidebar.selectbox("Model A", models, index=0)
    model_b = st.sidebar.selectbox("Model B", models, index=1 if len(models) > 1 else 0)
    if model_a == model_b:
        st.warning("Choose two different models.")
        return

    dataset_path = run_payload.get("dataset_path") or "data/labels/classification_corpus.jsonl"
    dataset_path = str((ROOT / dataset_path).resolve()) if not str(dataset_path).startswith("/") else dataset_path
    if not Path(dataset_path).exists():
        st.warning(f"Dataset path from run metadata not found: {dataset_path}. Falling back to classification corpus.")
        dataset_path = str(ROOT / "data/labels/classification_corpus.jsonl")

    dataset = cached_dataset(dataset_path)
    result_a = cached_resultset(str(resultsets_by_model[model_a]))
    result_b = cached_resultset(str(resultsets_by_model[model_b]))
    catalog_path = run_payload.get("model_catalog_path") or "config/model_catalog.json"
    catalog_abs = ROOT / catalog_path
    catalog = cached_json(str(catalog_abs)) if catalog_abs.exists() else {"models": {}}

    comparison = compare_two_models(dataset, result_a, result_b)
    metrics = comparison["metrics"]
    strict = comparison["strict_contract"]
    scored = certified_issues(dataset)
    unscored = uncertified_issues(dataset)
    issue_validation = validate_same_issue_set(result_a, result_b)

    render_header(run_payload, dataset_path, result_a, result_b, issue_validation)

    tab_overview, tab_side_by_side, tab_scored, tab_unscored, tab_ops, tab_dataset = st.tabs(
        [
            "Overview / Recommendation",
            "Side by Side",
            "Scored View",
            "Unscored View",
            "Operational Metrics",
            "Dataset Browser",
        ]
    )

    with tab_overview:
        render_overview(dataset, scored, unscored, result_a, result_b, metrics, strict, run_payload)
    with tab_side_by_side:
        render_side_by_side(dataset, result_a, result_b, metrics)
    with tab_scored:
        render_scored(result_a, result_b, metrics)
    with tab_unscored:
        render_unscored(dataset, result_a, result_b)
    with tab_ops:
        render_operational(result_a, result_b, catalog, run_payload)
    with tab_dataset:
        render_dataset_browser(dataset)


def render_new_comparison() -> None:
    st.subheader("New Comparison")
    st.write(
        "Run two selected models against the full classification corpus. "
        "Each issue is sent as its own inference request."
    )
    st.warning("This action calls paid DigitalOcean Serverless Inference models.")

    catalog_path = ROOT / "config/model_catalog.json"
    if not catalog_path.exists():
        st.error("Missing config/model_catalog.json. Build the runner model catalog first.")
        st.code(
            "python3 scripts/build_runner_model_catalog.py "
            "--metadata config/model_metadata.json "
            "--screening-pool config/screening_pool.csv "
            "--output config/model_catalog.json",
            language="bash",
        )
        return

    catalog = cached_json(str(catalog_path))
    model_ids = sorted((catalog.get("models") or {}).keys())
    if len(model_ids) < 2:
        st.error("Model catalog must contain at least two models.")
        return

    api_key_present = any(
        os.environ.get(name)
        for name in ("DIGITALOCEAN_SI_API_KEY", "DO_INFERENCE_API_KEY", "DIGITALOCEAN_TOKEN")
    )
    if not api_key_present:
        st.info(
            "No API key is visible to the app process. Set DIGITALOCEAN_SI_API_KEY, "
            "DO_INFERENCE_API_KEY, or DIGITALOCEAN_TOKEN before launching the container/app."
        )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    model_a = col_a.selectbox("Model A", model_ids, index=0)
    default_b = 1 if len(model_ids) > 1 else 0
    model_b = col_b.selectbox("Model B", model_ids, index=default_b)
    concurrency = col_c.number_input("Concurrency", min_value=1, max_value=32, value=2, step=1)

    st.caption(
        "Defaults: full corpus, temperature=0, timeout=45s, max_retries=1, "
        "max_output_tokens=2048, progress_interval=5."
    )

    active_run_id = st.session_state.get("active_eval_run_id")
    if active_run_id:
        render_run_progress(ROOT / "runs" / active_run_id)

    disabled = model_a == model_b or not api_key_present
    if model_a == model_b:
        st.warning("Choose two different models.")

    if st.button("Run full comparison", disabled=disabled, type="primary"):
        run_id = make_ui_run_id(model_a, model_b)
        launch_eval_run(run_id, model_a, model_b, int(concurrency))
        st.session_state["active_eval_run_id"] = run_id
        st.rerun()


def make_ui_run_id(model_a: str, model_b: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ui-{timestamp}-{safe_id(model_a)}-vs-{safe_id(model_b)}"


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:80]


def launch_eval_run(run_id: str, model_a: str, model_b: str, concurrency: int) -> None:
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_eval.py"),
        "--dataset",
        "data/labels/classification_corpus.jsonl",
        "--model-catalog",
        "config/model_catalog.json",
        "--prompt",
        "config/prompts/classification_template.txt",
        "--models",
        f"{model_a},{model_b}",
        "--output-dir",
        "runs",
        "--run-id",
        run_id,
        "--concurrency",
        str(concurrency),
        "--timeout-seconds",
        "45",
        "--max-retries",
        "1",
        "--temperature",
        "0",
        "--max-output-tokens",
        "2048",
        "--progress-interval",
        "5",
        "--all",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def render_run_progress(run_dir: Path) -> None:
    run_json = run_dir / "run.json"
    run_log = run_dir / "run.log"
    run_payload = load_json(run_json) if run_json.exists() else {}
    status = run_payload.get("status", "starting")
    model_ids = run_payload.get("model_ids") or []
    issue_count = run_payload.get("issue_count") or 0
    progress_rows = load_progress_rows(run_dir)

    completed_total = sum(row.get("completed_count", 0) for row in progress_rows)
    expected_total = issue_count * len(model_ids) if issue_count and model_ids else None
    progress = completed_total / expected_total if expected_total else 0.0

    st.subheader("Run Progress")
    st.write(f"Run ID: `{run_dir.name}`")
    st.progress(min(max(progress, 0.0), 1.0), text=f"{completed_total}/{expected_total or '?'} calls completed")

    if progress_rows:
        dataframe(pd.DataFrame(progress_rows))
    else:
        st.caption("Waiting for progress files...")

    if run_log.exists():
        with st.expander("Runner log", expanded=False):
            st.code(tail_text(run_log, max_lines=80))

    if status == "completed":
        st.success("Run completed. Select this run from the sidebar to inspect the comparison.")
        st.session_state.pop("active_eval_run_id", None)
    elif status == "failed":
        st.error("Run failed. Open the runner log for details.")
        st.session_state.pop("active_eval_run_id", None)
    else:
        time.sleep(2)
        st.rerun()


def run_is_active(run_dir: Path, run_payload: dict[str, Any]) -> bool:
    """Return true when a selected persisted run should show the progress view."""

    status = str(run_payload.get("status") or "").lower()
    if status in {"starting", "running"}:
        return True
    if status in {"completed", "failed"}:
        return False

    progress_rows = load_progress_rows(run_dir)
    if not progress_rows:
        return False
    return any(str(row.get("status") or "").lower() not in {"completed", "failed"} for row in progress_rows)


def load_progress_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "progress").glob("*.json")):
        try:
            row = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "model_id": row.get("model_id"),
                "status": row.get("status"),
                "completed_count": row.get("completed_count"),
                "total_count": row.get("total_count"),
                "ok_count": row.get("ok_count"),
                "error_count": row.get("error_count"),
                "requests_per_second": row.get("requests_per_second"),
                "eta_seconds": row.get("eta_seconds"),
                "updated_at": row.get("updated_at"),
            }
        )
    return rows


def tail_text(path: Path, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def effective_resultset_index(run_dir: Path) -> dict[str, Path]:
    """Return model resultsets for a run, honoring screening summary merge metadata.

    Model screening sometimes reruns only a subset of models, then summarizes with
    baseline resultsets from earlier run directories. When screening_summary.json is
    present, its resultset_path values represent the effective comparison pool.
    For ordinary pilot/final runs, local runs/<run-id>/results/*.json is used.
    """

    local = {path.stem: path for path in resultset_paths(run_dir)}
    summary_path = run_dir / "screening_summary.json"
    if not summary_path.exists():
        return local

    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return local

    merged: dict[str, Path] = {}
    for row in summary.get("models", []):
        model_id = row.get("model_id")
        resultset_path = row.get("resultset_path")
        if not model_id or not resultset_path:
            continue
        path = Path(resultset_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            merged[str(model_id)] = path

    return merged or local


def render_header(
    run_payload: dict[str, Any],
    dataset_path: str,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    issue_validation: dict[str, Any],
) -> None:
    st.subheader("Run Context")
    cols = st.columns(4)
    cols[0].metric("Run ID", run_payload.get("run_id", "N/A"))
    cols[1].metric("Issue Count", run_payload.get("issue_count", "N/A"))
    cols[2].metric("Model A", result_a["model_id"])
    cols[3].metric("Model B", result_b["model_id"])

    with st.expander("Run metadata", expanded=False):
        st.write(f"Dataset: `{dataset_path}`")
        fields = [
            "prompt_source",
            "model_catalog_path",
            "concurrency",
            "timeout_seconds",
            "max_retries",
            "temperature",
            "max_output_tokens",
            "created_at",
            "completed_at",
            "wall_clock_ms",
        ]
        dataframe(pd.DataFrame([{"field": key, "value": run_payload.get(key)} for key in fields]))

    if issue_validation["a_missing_from_b"] or issue_validation["b_missing_from_a"]:
        st.warning(
            "Selected models do not have identical issue sets. "
            f"Shared={issue_validation['shared_count']}, "
            f"A only={len(issue_validation['a_missing_from_b'])}, "
            f"B only={len(issue_validation['b_missing_from_a'])}."
        )

    run_id = str(run_payload.get("run_id", ""))
    dataset = str(run_payload.get("dataset_path", ""))
    if "screening" in run_id or "screening_corpus" in dataset:
        st.info("Screening runs are for operational screening, not final quality ranking.")


def render_overview(
    dataset: dict[str, Any],
    scored: list[dict[str, Any]],
    unscored: list[dict[str, Any]],
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    strict: dict[str, dict[str, Any]],
    run_payload: dict[str, Any],
) -> None:
    st.write(
        "This page compares two selected models on the same doctl issue corpus. "
        "Quality metrics are computed on the certified scored subset; unscored issues are used for agreement and behavior analysis."
    )
    cols = st.columns(3)
    cols[0].metric("Total Issues", len(dataset["issues"]))
    cols[1].metric("Scored / Certified", len(scored))
    cols[2].metric("Unscored / Review", len(unscored))
    if len(scored) < 30:
        st.warning("This run has fewer than 30 scored issues. Scored metrics are directional.")

    distribution = scored_label_distribution(dataset)
    low_support = distribution[distribution["count"] < 10]
    if not low_support.empty:
        st.warning("Low-support classes are reported for transparency but should not dominate model selection.")
    st.caption("Scored label support")
    st.markdown(compact_label_distribution(distribution))

    st.subheader("Executive Comparison")
    executive = executive_comparison_table(result_a, result_b, metrics, strict)
    render_overview_leader_cards(executive)
    dataframe(focused_executive_table(executive))

    st.subheader("Tradeoff Snapshot")
    st.write("The UI does not auto-pick a winner; use these signals to choose default and fallback.")
    for note in decision_notes(executive):
        st.markdown(f"- {note}")


def compact_label_distribution(distribution: pd.DataFrame) -> str:
    parts = [
        f"`{row['label']}` {int(row['count'])}"
        for _, row in distribution.iterrows()
    ]
    return " · ".join(parts)


def render_overview_leader_cards(executive: pd.DataFrame) -> None:
    rows = executive.set_index("model_id").to_dict("index")
    model_ids = list(rows)
    if len(model_ids) < 2:
        return
    model_a, model_b = model_ids[0], model_ids[1]
    cards = st.columns(4)
    card_specs = [
        ("Quality", "end_to_end_scored_accuracy", True, False, "%"),
        ("Cost", "total_run_cost_usd", False, True, ""),
        ("Latency", "p95_latency_ms", False, False, "ms"),
        ("Reliability", "error_rate", False, False, "%"),
    ]
    for column, (title, key, higher_is_better, money, unit) in zip(cards, card_specs, strict=True):
        winner = metric_winner(rows, model_a, model_b, key, higher_is_better)
        value = rows.get(winner, {}).get(key) if winner else None
        column.metric(
            f"{title} leader",
            winner or "N/A",
            format_metric_value(value, money=money, unit=unit) if isinstance(value, int | float) else None,
        )


def focused_executive_table(executive: pd.DataFrame) -> pd.DataFrame:
    rows = executive.set_index("model_id").to_dict("index")
    model_ids = list(rows)
    if len(model_ids) < 2:
        return executive
    model_a, model_b = model_ids[0], model_ids[1]
    metric_specs = [
        ("End-to-end scored accuracy", "end_to_end_scored_accuracy", True, False, "%"),
        ("Evaluated accuracy", "evaluated_accuracy", True, False, "%"),
        ("Stable macro F1", "stable_macro_f1", True, False, "%"),
        ("Bug recall", "bug_recall", True, False, "%"),
        ("Security recall", "security_recall", True, False, "%"),
        ("Strict output valid rate", "strict_output_valid_rate", True, False, "%"),
        ("Total run cost", "total_run_cost_usd", False, True, ""),
        ("Cost per correct", "cost_per_correct_usd", False, True, ""),
        ("p95 latency", "p95_latency_ms", False, False, "ms"),
        ("Error rate", "error_rate", False, False, "%"),
    ]
    table_rows = []
    for label, key, higher_is_better, money, unit in metric_specs:
        value_a = rows[model_a].get(key)
        value_b = rows[model_b].get(key)
        table_rows.append(
            {
                "metric": label,
                model_a: format_metric_value(value_a, money=money, unit=unit),
                model_b: format_metric_value(value_b, money=money, unit=unit),
                "better": metric_winner(rows, model_a, model_b, key, higher_is_better) or "",
            }
        )
    return pd.DataFrame(table_rows)


def metric_winner(
    rows: dict[str, dict[str, Any]],
    model_a: str,
    model_b: str,
    key: str,
    higher_is_better: bool,
) -> str | None:
    value_a = rows[model_a].get(key)
    value_b = rows[model_b].get(key)
    if not isinstance(value_a, int | float) or not isinstance(value_b, int | float):
        return None
    if value_a == value_b:
        return "tie"
    return model_a if (value_a > value_b) == higher_is_better else model_b


def decision_notes(executive: pd.DataFrame) -> list[str]:
    if len(executive) < 2:
        return ["Select two resultsets to compare."]
    rows = executive.set_index("model_id").to_dict("index")
    model_ids = list(rows)
    a, b = model_ids[0], model_ids[1]
    notes = []
    notes.append(compare_metric(rows, a, b, "end_to_end_scored_accuracy", "end-to-end scored accuracy", higher_is_better=True, unit="%"))
    notes.append(compare_metric(rows, a, b, "total_run_cost_usd", "total run cost", higher_is_better=False, money=True))
    notes.append(compare_metric(rows, a, b, "p95_latency_ms", "p95 latency", higher_is_better=False, unit="ms"))
    notes.append(compare_metric(rows, a, b, "bug_recall", "bug recall", higher_is_better=True, unit="%"))
    notes.append(compare_metric(rows, a, b, "strict_output_valid_rate", "strict output validity", higher_is_better=True, unit="%"))
    return [note for note in notes if note]


def compare_metric(
    rows: dict[str, dict[str, Any]],
    model_a: str,
    model_b: str,
    key: str,
    label: str,
    higher_is_better: bool,
    money: bool = False,
    unit: str = "",
) -> str:
    value_a = rows[model_a].get(key)
    value_b = rows[model_b].get(key)
    if not isinstance(value_a, int | float) or not isinstance(value_b, int | float):
        return ""
    if value_a == value_b:
        return f"{label}: tied at {format_metric_value(value_a, money, unit)}."
    winner = model_a if (value_a > value_b) == higher_is_better else model_b
    loser = model_b if winner == model_a else model_a
    winner_value = value_a if winner == model_a else value_b
    loser_value = value_b if winner == model_a else value_a
    return (
        f"{label}: `{winner}` leads "
        f"({format_metric_value(winner_value, money, unit)} vs {format_metric_value(loser_value, money, unit)} for `{loser}`)."
    )


def format_metric_value(value: Any, money: bool = False, unit: str = "") -> str:
    if not isinstance(value, int | float):
        return "N/A"
    if money:
        return f"${value:.6f}"
    if unit == "%":
        return f"{value * 100:.1f}%"
    if unit:
        return f"{value:,.1f}{unit}"
    return f"{value:.3f}"


def render_scored(result_a: dict[str, Any], result_b: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> None:
    st.write("Overall accuracy is a headline, not the decision metric. Per-class recall shows which true issue types a model misses.")
    model_a = result_a["model_id"]
    model_b = result_b["model_id"]
    table = side_by_side_per_class_table(model_a, model_b, metrics[model_a], metrics[model_b])
    dataframe(table)
    st.write("Confusion matrices and concrete case drill-downs are available in Side by Side.")


def render_dataset_browser(dataset: dict[str, Any]) -> None:
    st.write("Browse the issue corpus used by the selected run.")
    raw_index = cached_raw_issue_index(str(ROOT / "data/doctl_issues.json"))
    table = dataset_browser_table(dataset, raw_index)

    filters = st.columns([2, 2, 2, 2, 1])
    text = filters[0].text_input("Search title/details")
    split = filters[1].selectbox("Split", ["all", *sorted(table["Split"].dropna().unique())])
    truth = filters[2].selectbox("Ground truth", ["all", *sorted(table["Ground Truth Category"].dropna().unique())])
    maintainer = filters[3].text_input("Maintainer label contains")
    min_comments = filters[4].number_input("Min comments", min_value=0, value=0, step=1)

    filtered = table.copy()
    if text:
        needle = text.lower()
        filtered = filtered[
            filtered["Title"].str.lower().str.contains(needle, na=False)
            | filtered["Details"].str.lower().str.contains(needle, na=False)
        ]
    if split != "all":
        filtered = filtered[filtered["Split"] == split]
    if truth != "all":
        filtered = filtered[filtered["Ground Truth Category"] == truth]
    if maintainer:
        filtered = filtered[
            filtered["Maintainer Labels"].str.lower().str.contains(maintainer.lower(), na=False)
        ]
    if min_comments:
        filtered = filtered[filtered["Comments"] >= min_comments]

    st.caption(f"Showing {len(filtered)} of {len(table)} issues.")
    visible_columns = ["Title", "Details", "Maintainer Labels", "Comments", "Ground Truth Category", "URL"]
    display_rows = filtered[["Issue Number", *visible_columns]].reset_index(drop=True)
    event = st.dataframe(
        arrow_safe_dataframe(display_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Issue Number": None,
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Details": st.column_config.TextColumn("Details", width="large"),
            "Maintainer Labels": st.column_config.TextColumn("Maintainer Labels"),
            "Comments": st.column_config.NumberColumn("Comments"),
            "Ground Truth Category": st.column_config.TextColumn("Ground Truth Category"),
            "URL": st.column_config.LinkColumn("URL", display_text="Open"),
        },
        column_order=visible_columns,
        height=620,
        key="dataset_browser_table",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_issue_number = selected_issue_from_table_event(event, display_rows)
    render_issue_review_panel(selected_issue_number, raw_index)


def dataset_browser_table(dataset: dict[str, Any], raw_index: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for issue in dataset.get("issues", []):
        issue_number = int(issue["issue_number"])
        raw = raw_index.get(issue_number, {})
        labels = raw.get("labels")
        if labels is None:
            labels = issue.get("maintainer_labels") or []
            label_names = [str(label) for label in labels]
        else:
            label_names = [str(label.get("name", "")) for label in labels if label.get("name")]
        ground_truth = issue.get("ground_truth", {})
        label = ground_truth.get("label")
        rows.append(
            {
                "Issue Number": issue_number,
                "Title": f"#{issue_number}: {issue.get('title') or ''}",
                "Details": issue_detail_text(issue),
                "Maintainer Labels": ", ".join(label_names),
                "Comments": int(raw.get("comments") or 0),
                "Ground Truth Category": label or "",
                "URL": issue.get("html_url") or raw.get("html_url") or "",
                "Split": ground_truth.get("status", "uncertified"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("Issue Number")


def selected_issue_from_table_event(event: Any, display_rows: pd.DataFrame) -> int | None:
    if display_rows.empty:
        return None
    selected_rows: list[int] = []
    if event is not None:
        selected_rows = getattr(getattr(event, "selection", None), "rows", None) or []
        if not selected_rows and isinstance(event, dict):
            selected_rows = event.get("selection", {}).get("rows", [])
    if selected_rows:
        row_index = int(selected_rows[0])
        if 0 <= row_index < len(display_rows):
            return int(display_rows.iloc[row_index]["Issue Number"])
    return int(display_rows.iloc[0]["Issue Number"])


def render_issue_review_panel(issue_number: int | None, raw_index: dict[int, dict[str, Any]]) -> None:
    if issue_number is None:
        return
    st.subheader("Selected Issue Review")
    raw = raw_index.get(issue_number, {})
    if not raw:
        st.info("Raw issue snapshot not found for this issue.")
        return

    st.markdown(f"**Issue #{issue_number}: {raw.get('title') or ''}**")
    st.write(raw.get("html_url") or "")
    st.caption("Issue body")
    st.markdown(
        f"<div class='issue-body-box'>{html_escape(raw.get('body') or '')}</div>",
        unsafe_allow_html=True,
    )

    comments = raw.get("comment_details") or []
    comment_count = int(raw.get("comments") or 0)
    if not comments and comment_count:
        st.info(
            f"This issue has {comment_count} GitHub comments, but comment bodies are not in the local snapshot yet. "
            "Refresh data with `python3 scripts/fetch_github_issues.py --include-comments`."
        )
        return
    if not comments:
        st.caption("No comments.")
        return

    st.caption(f"{len(comments)} comments")
    for index, comment in enumerate(comments, start=1):
        author = comment.get("user_login") or "unknown"
        created = comment.get("created_at") or ""
        with st.expander(f"Comment {index} by {author} at {created}", expanded=index <= 2):
            st.write(comment.get("html_url") or "")
            st.markdown(comment.get("body") or "")


def issue_detail_text(issue: dict[str, Any], limit: int = 420) -> str:
    body = (issue.get("body") or "").replace("\r", " ").replace("\n", " ")
    body = " ".join(body.split())
    if len(body) > limit:
        return body[: limit - 3] + "..."
    return body


def render_side_by_side(
    dataset: dict[str, Any],
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
) -> None:
    st.write("Confusion matrix: rows are certified ground truth labels; columns are model-predicted labels. Diagonal cells are correct classifications.")
    normalized = st.checkbox("Show row-normalized confusion matrices", value=False)
    col_a, col_b = st.columns(2)
    with col_a:
        render_selectable_confusion_matrix(result_a["model_id"], metrics[result_a["model_id"]], normalized)
    with col_b:
        render_selectable_confusion_matrix(result_b["model_id"], metrics[result_b["model_id"]], normalized)

    cases = scored_case_table(dataset, result_a, result_b)
    filtered = apply_confusion_cell_filter(cases, result_a["model_id"], result_b["model_id"])
    render_selectable_case_list(filtered, result_a["model_id"], result_b["model_id"], include_truth=True)


def render_selectable_confusion_matrix(model_id: str, metrics: dict[str, Any], normalized: bool) -> None:
    st.markdown(f"### {model_id}")
    st.caption("Click a number to show the represented cases below.")
    matrix = confusion_matrix_dataframe(metrics, normalized=normalized).reindex(index=LABELS, columns=LABELS).fillna(0)
    header = st.columns([1.35, *([1] * len(LABELS))])
    header[0].markdown("**Actual \\ Predicted**")
    for index, label in enumerate(LABELS, start=1):
        header[index].markdown(f"**{short_label(label)}**")

    for actual in LABELS:
        row = st.columns([1.35, *([1] * len(LABELS))])
        row[0].markdown(f"**{actual}**")
        for index, predicted in enumerate(LABELS, start=1):
            value = matrix.loc[actual, predicted]
            label = f"{float(value):.0%}" if normalized else str(int(value))
            if row[index].button(label, key=f"cm-{model_id}-{actual}-{predicted}", use_container_width=True):
                st.session_state["confusion_cell_filter"] = {
                    "model_id": model_id,
                    "actual": actual,
                    "predicted": predicted,
                }


def short_label(label: str) -> str:
    return {
        "enhancement": "enhance",
        "documentation": "docs",
        "question": "question",
        "security": "security",
        "other": "other",
        "bug": "bug",
    }.get(label, label)


def apply_confusion_cell_filter(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    cell = st.session_state.get("confusion_cell_filter")
    if not cell:
        return df
    model_id = cell.get("model_id")
    actual = cell.get("actual")
    predicted = cell.get("predicted")
    if model_id not in {model_a, model_b} or not actual or not predicted:
        return df
    label_column = f"{model_id} label"
    if label_column not in df.columns:
        return df

    st.info(f"Showing confusion cell: `{model_id}` actual `{actual}` predicted `{predicted}`.")
    if st.button("Clear confusion-cell filter"):
        st.session_state.pop("confusion_cell_filter", None)
        st.rerun()
    return df[(df["ground_truth"] == actual) & (df[label_column] == predicted)]


def render_selectable_case_list(df: pd.DataFrame, model_a: str, model_b: str, include_truth: bool) -> None:
    if df.empty:
        st.info("No cases match the current selection.")
        return
    st.caption(f"Showing {len(df)} matching cases. Select one row to inspect details.")
    display_columns = display_case_columns(model_a, model_b, include_truth=include_truth)
    display_rows = df[display_columns].reset_index(drop=True)
    event = st.dataframe(
        arrow_safe_dataframe(display_rows),
        width="stretch",
        hide_index=True,
        key=f"case-list-{model_a}-{model_b}-{include_truth}",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_index = selected_row_index(event, display_rows)
    if selected_index is None:
        selected_index = 0
    selected_row = df.reset_index(drop=True).iloc[selected_index]
    render_case_detail(selected_row, model_a, model_b)


def selected_row_index(event: Any, display_rows: pd.DataFrame) -> int | None:
    if display_rows.empty:
        return None
    selected_rows: list[int] = []
    if event is not None:
        selected_rows = getattr(getattr(event, "selection", None), "rows", None) or []
        if not selected_rows and isinstance(event, dict):
            selected_rows = event.get("selection", {}).get("rows", [])
    if selected_rows:
        row_index = int(selected_rows[0])
        if 0 <= row_index < len(display_rows):
            return row_index
    return None


def apply_scored_filters(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    cols = st.columns(4)
    truth = cols[0].selectbox("Ground truth", ["all", *sorted(df["ground_truth"].dropna().unique())])
    pred_a = cols[1].selectbox("Model A predicted", ["all", *sorted(df[f"{model_a} label"].dropna().unique())])
    pred_b = cols[2].selectbox("Model B predicted", ["all", *sorted(df[f"{model_b} label"].dropna().unique())])
    outcome = cols[3].selectbox(
        "Outcome",
        ["all", "A correct / B wrong", "B correct / A wrong", "both wrong", "both correct", "models disagree"],
    )
    critical = st.checkbox("Critical labels only: bug/security", value=False)
    errors = st.checkbox("Errors/invalid only", value=False)
    out = df.copy()
    if truth != "all":
        out = out[out["ground_truth"] == truth]
    if pred_a != "all":
        out = out[out[f"{model_a} label"] == pred_a]
    if pred_b != "all":
        out = out[out[f"{model_b} label"] == pred_b]
    if outcome == "A correct / B wrong":
        out = out[(out["A correct"] == True) & (out["B correct"] == False)]
    elif outcome == "B correct / A wrong":
        out = out[(out["B correct"] == True) & (out["A correct"] == False)]
    elif outcome == "both wrong":
        out = out[(out["A correct"] == False) & (out["B correct"] == False)]
    elif outcome == "both correct":
        out = out[(out["A correct"] == True) & (out["B correct"] == True)]
    elif outcome == "models disagree":
        out = out[out["models_disagree"]]
    if critical:
        out = out[out["critical_truth"]]
    if errors:
        out = out[(out["A status"] != "ok") | (out["B status"] != "ok")]
    return out


def render_unscored(dataset: dict[str, Any], result_a: dict[str, Any], result_b: dict[str, Any]) -> None:
    st.warning("Unscored issues do not have certified ground truth. Agreement and distribution are diagnostic only; they are not accuracy metrics.")
    analysis = unscored_analysis([result_a, result_b], uncertified_issues(dataset))
    cols = st.columns(5)
    cols[0].metric("Unscored issues", analysis.get("unscored_issue_count"))
    cols[1].metric("Comparable", analysis.get("comparable_issue_count"))
    cols[2].metric("Agreement", analysis.get("agreement_count"))
    cols[3].metric("Agreement rate", format_number(analysis.get("agreement_rate")))
    cols[4].metric("Disagreements", len(analysis.get("disagreements", [])))

    st.subheader("Prediction Distribution")
    dataframe(prediction_distribution_table(dataset, result_a, result_b))

    cases = unscored_case_table(dataset, result_a, result_b)
    if cases.empty:
        st.info("This dataset has no unscored/review issues to inspect.")
        return
    filtered = apply_unscored_filters(cases, result_a["model_id"], result_b["model_id"])
    dataframe(filtered[display_case_columns(result_a["model_id"], result_b["model_id"], include_truth=False)])
    render_case_expanders(filtered, result_a["model_id"], result_b["model_id"])


def apply_unscored_filters(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    cols = st.columns(4)
    only_disagreements = cols[0].checkbox("Only disagreements", value=False)
    pred_a = cols[1].selectbox("Model A label", ["all", *sorted(df[f"{model_a} label"].dropna().unique())], key="unscored_a")
    pred_b = cols[2].selectbox("Model B label", ["all", *sorted(df[f"{model_b} label"].dropna().unique())], key="unscored_b")
    special = cols[3].selectbox(
        "Triage filter",
        ["all", "A predicted security", "B predicted security", "A predicted other", "B predicted other", "error/missing only", "security keyword"],
    )
    out = df.copy()
    if only_disagreements:
        out = out[out["models_disagree"]]
    if pred_a != "all":
        out = out[out[f"{model_a} label"] == pred_a]
    if pred_b != "all":
        out = out[out[f"{model_b} label"] == pred_b]
    if special == "A predicted security":
        out = out[out[f"{model_a} label"] == "security"]
    elif special == "B predicted security":
        out = out[out[f"{model_b} label"] == "security"]
    elif special == "A predicted other":
        out = out[out[f"{model_a} label"] == "other"]
    elif special == "B predicted other":
        out = out[out[f"{model_b} label"] == "other"]
    elif special == "error/missing only":
        out = out[(out["A status"] != "ok") | (out["B status"] != "ok")]
    elif special == "security keyword":
        out = out[out["security_keyword"]]
    return out


def display_case_columns(model_a: str, model_b: str, include_truth: bool) -> list[str]:
    cols = ["issue_number", "title", "html_url"]
    if include_truth:
        cols.append("ground_truth")
    return cols + [
        f"{model_a} label",
        f"{model_b} label",
        "A correct" if include_truth else "agree",
        "B correct" if include_truth else "models_disagree",
        "A latency_ms",
        "B latency_ms",
        "A cost_usd",
        "B cost_usd",
    ]


def render_case_expanders(df: pd.DataFrame, model_a: str, model_b: str) -> None:
    st.caption(f"Showing details for first {min(len(df), 25)} filtered rows.")
    for _, row in df.head(25).iterrows():
        with st.expander(f"#{row['issue_number']} {row['title']}"):
            st.write(row.get("html_url"))
            st.caption("Issue body excerpt")
            st.markdown(
                f"<div class='issue-body-box'>{html_escape(row.get('body_excerpt') or '')}</div>",
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{model_a}**")
                render_model_raw_output(row.get("A raw_output") or row.get("A error"))
                st.write("Rationale:", row.get("A rationale") or "N/A")
            with col_b:
                st.markdown(f"**{model_b}**")
                render_model_raw_output(row.get("B raw_output") or row.get("B error"))
                st.write("Rationale:", row.get("B rationale") or "N/A")


def render_case_detail(row: pd.Series, model_a: str, model_b: str) -> None:
    st.subheader("Selected Case Detail")
    st.markdown(f"**#{row['issue_number']} {row['title']}**")
    st.write(row.get("html_url"))
    st.caption("Issue body excerpt")
    st.markdown(
        f"<div class='issue-body-box'>{html_escape(row.get('body_excerpt') or '')}</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{model_a}**")
        st.write("Predicted label:", row.get(f"{model_a} label") or "N/A")
        st.write("Correct:", row.get("A correct"))
        render_model_raw_output(row.get("A raw_output") or row.get("A error"))
        st.write("Rationale:", row.get("A rationale") or "N/A")
    with col_b:
        st.markdown(f"**{model_b}**")
        st.write("Predicted label:", row.get(f"{model_b} label") or "N/A")
        st.write("Correct:", row.get("B correct"))
        render_model_raw_output(row.get("B raw_output") or row.get("B error"))
        st.write("Rationale:", row.get("B rationale") or "N/A")


def render_model_raw_output(raw: Any) -> None:
    st.caption("Raw model output")
    if raw is None or raw == "":
        st.code("N/A")
        return
    if isinstance(raw, str):
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            st.code(text, language="text")
            return
        st.json(parsed, expanded=True)
        return
    if isinstance(raw, dict | list):
        st.json(raw, expanded=True)
    else:
        st.code(str(raw), language="text")


def render_operational(result_a: dict[str, Any], result_b: dict[str, Any], catalog: dict[str, Any], run_payload: dict[str, Any]) -> None:
    st.subheader("Per-run Operational Metrics")
    dataframe(operational_summary_table([result_a, result_b]))

    st.subheader("Cost Trace")
    dataframe(token_cost_trace([result_a, result_b], catalog))
    st.code(
        "cost_usd = prompt_tokens / 1_000_000 * input_price_per_1m + "
        "completion_tokens / 1_000_000 * output_price_per_1m"
    )

    st.subheader("Run Config")
    config_fields = [
        "concurrency",
        "timeout_seconds",
        "max_retries",
        "rate_limit_retry_policy",
        "max_output_tokens",
        "temperature",
        "prompt_source",
    ]
    dataframe(pd.DataFrame([{"field": key, "value": run_payload.get(key)} for key in config_fields]))

    st.subheader("Production Handling Notes")
    st.markdown(
        """
- Invalid output -> retry with stricter prompt or route to fallback.
- Timeout/retry exhausted -> route to fallback.
- Security keyword issue -> route to stronger model or human review.
- Model disagreement on critical labels -> human review or stronger model.
- Cheap model predicts `other` -> consider fallback if business impact is high.
"""
    )


if __name__ == "__main__":
    main()
