"""
Parse OhioT1DM XML files and output data quality statistics.
"""
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_xml_file(filepath):
    """Parse a single OhioT1DM XML file and return structured data."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    patient_id = root.attrib.get("id", "unknown")

    # Parse glucose events
    glucose_events = []
    for child in root:
        if child.tag == "glucose_level":
            for event in child:
                ts = event.attrib.get("ts", "")
                value = event.attrib.get("value", "")
                if ts and value:
                    try:
                        dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
                        glucose_events.append({
                            "timestamp": dt,
                            "glucose": float(value)
                        })
                    except (ValueError, TypeError):
                        pass

    # Parse bolus events.  Preserve the exact begin/end timestamps: the new
    # one-step MDP treats ts_begin as an observed delivery-episode start and
    # allocates extended delivery over [ts_begin, ts_end].
    bolus_events = []
    for child in root:
        if child.tag == "bolus":
            for event in child:
                ts_begin = event.attrib.get("ts_begin", "")
                ts_end = event.attrib.get("ts_end", "")
                dose = event.attrib.get("dose", "")
                bolus_type = event.attrib.get("type", "normal")
                if ts_begin and dose:
                    try:
                        dt_begin = datetime.strptime(ts_begin, "%d-%m-%Y %H:%M:%S")
                        dt_end = datetime.strptime(ts_end, "%d-%m-%Y %H:%M:%S") if ts_end else dt_begin
                        bolus_events.append({
                            "ts_begin": dt_begin,
                            "ts_end": dt_end,
                            "dose": float(dose),
                            "type": bolus_type,
                            "bwz_carb_input": _safe_float(event.attrib.get("bwz_carb_input")),
                        })
                    except (ValueError, TypeError):
                        pass

    # Parse meal events
    meal_events = []
    for child in root:
        if child.tag == "meal":
            for event in child:
                ts = event.attrib.get("ts", "")
                carbs = event.attrib.get("carbs", "")
                meal_type = event.attrib.get("type", "")
                if ts and carbs:
                    try:
                        dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
                        meal_events.append({
                            "timestamp": dt,
                            "carbs": float(carbs),
                            "type": meal_type
                        })
                    except (ValueError, TypeError):
                        pass

    # Parse basal events
    basal_events = []
    for child in root:
        if child.tag == "basal":
            for event in child:
                ts = event.attrib.get("ts", "")
                value = event.attrib.get("value", "")
                if ts and value:
                    try:
                        dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
                        basal_events.append({
                            "timestamp": dt,
                            "rate": float(value)
                        })
                    except (ValueError, TypeError):
                        pass

    # Temporary basal overrides the scheduled basal on [ts_begin, ts_end).
    temp_basal_events = []
    for child in root:
        if child.tag == "temp_basal":
            for event in child:
                ts_begin = event.attrib.get("ts_begin", "")
                ts_end = event.attrib.get("ts_end", "")
                value = event.attrib.get("value", "")
                if ts_begin and ts_end and value:
                    try:
                        temp_basal_events.append({
                            "ts_begin": datetime.strptime(ts_begin, "%d-%m-%Y %H:%M:%S"),
                            "ts_end": datetime.strptime(ts_end, "%d-%m-%Y %H:%M:%S"),
                            "rate": float(value),
                        })
                    except (ValueError, TypeError):
                        pass

    # Events below are not part of the frozen state vector.  They are retained
    # so the action-timing forensic audit can identify information that appeared
    # after S_t was frozen but before a bolus in the same five-minute cell.
    context_events = []
    point_event_tags = {
        "finger_stick": "finger_stick",
        "hypo_event": "hypo_event",
        "exercise": "exercise",
        "stressors": "stressor",
    }
    interval_event_tags = {
        "work": "work",
        "illness": "illness",
        "sleep": "sleep",
    }
    for child in root:
        if child.tag in point_event_tags:
            for event in child:
                ts = event.attrib.get("ts", "")
                if not ts:
                    continue
                try:
                    context_events.append({
                        "timestamp": datetime.strptime(ts, "%d-%m-%Y %H:%M:%S"),
                        "kind": point_event_tags[child.tag],
                        "phase": "point",
                    })
                except (ValueError, TypeError):
                    pass
        elif child.tag in interval_event_tags:
            for event in child:
                ts_begin = event.attrib.get("ts_begin", "")
                ts_end = event.attrib.get("ts_end", "")
                for raw_ts, phase in ((ts_begin, "begin"), (ts_end, "end")):
                    if not raw_ts:
                        continue
                    try:
                        context_events.append({
                            "timestamp": datetime.strptime(raw_ts, "%d-%m-%Y %H:%M:%S"),
                            "kind": interval_event_tags[child.tag],
                            "phase": phase,
                        })
                    except (ValueError, TypeError):
                        pass

    return {
        "patient_id": patient_id,
        "glucose": glucose_events,
        "bolus": bolus_events,
        "meal": meal_events,
        "basal": basal_events,
        "temp_basal": temp_basal_events,
        "context_events": context_events,
    }


def _safe_float(value):
    """Return a finite float when XML contains an optional numeric field."""
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def compute_data_quality(patient_data, resample_min=30):
    """Compute data quality metrics for a patient."""
    pid = patient_data["patient_id"]
    glucose = patient_data["glucose"]
    bolus = patient_data["bolus"]
    meal = patient_data["meal"]
    basal = patient_data["basal"]

    stats = {"patient_id": pid}
    stats["n_glucose"] = len(glucose)
    stats["n_bolus"] = len(bolus)
    stats["n_meal"] = len(meal)
    stats["n_basal"] = len(basal)

    if len(glucose) == 0:
        stats["start_time"] = None
        stats["end_time"] = None
        stats["duration_days"] = 0
        stats["n_windows_30min"] = 0
        stats["missing_rate"] = 1.0
        stats["bolus_action_ratio"] = 0.0
        stats["usable"] = False
        stats["reason"] = "No glucose data"
        return stats

    # Sort glucose by timestamp
    glucose_sorted = sorted(glucose, key=lambda x: x["timestamp"])
    stats["start_time"] = glucose_sorted[0]["timestamp"]
    stats["end_time"] = glucose_sorted[-1]["timestamp"]
    stats["duration_days"] = (stats["end_time"] - stats["start_time"]).total_seconds() / 86400.0

    # Expected 5-min intervals
    expected_5min_count = int(stats["duration_days"] * 24 * 12)  # 12 readings per hour
    if expected_5min_count > 0:
        stats["missing_rate"] = max(0.0, 1.0 - stats["n_glucose"] / expected_5min_count)
    else:
        stats["missing_rate"] = 1.0

    # 30-min window count (6 original readings per window)
    stats["n_windows_30min"] = stats["n_glucose"] // 6

    # Bolus action ratio (approximate, based on windows that would have bolus)
    if stats["n_windows_30min"] > 0:
        # We'll count this properly after resampling
        stats["bolus_action_ratio"] = "TBD"  # placeholder
    else:
        stats["bolus_action_ratio"] = 0.0

    # Check usability
    stats["usable"] = True
    reasons = []

    if stats["n_glucose"] < 100:
        reasons.append("Too few glucose readings (<100)")
        stats["usable"] = False
    if stats["n_windows_30min"] < 10:
        reasons.append("Too few 30min windows (<10)")
        stats["usable"] = False
    if stats["missing_rate"] > 0.5:
        reasons.append(f"High missing rate ({stats['missing_rate']:.2f})")
        stats["usable"] = False

    stats["reason"] = "; ".join(reasons) if reasons else "OK"

    return stats


def parse_all_patients(train_dir, test_dir):
    """Parse all training and testing XML files and return quality summary."""
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train_files = sorted(train_dir.glob("*-ws-training.xml"))
    test_files = sorted(test_dir.glob("*-ws-testing.xml"))

    all_stats = []

    print("=" * 80)
    print("PARSING TRAINING DATA")
    print("=" * 80)
    for fpath in train_files:
        print(f"  Parsing {fpath.name}...")
        data = parse_xml_file(fpath)
        stats = compute_data_quality(data)
        stats["split"] = "training"
        all_stats.append(stats)
        print(f"    Patient {stats['patient_id']}: "
              f"glucose={stats['n_glucose']}, bolus={stats['n_bolus']}, "
              f"meal={stats['n_meal']}, basal={stats['n_basal']}, "
              f"windows={stats['n_windows_30min']}, "
              f"missing={stats['missing_rate']:.2%}, "
              f"usable={stats['usable']}")

    print("\n" + "=" * 80)
    print("PARSING TESTING DATA")
    print("=" * 80)
    for fpath in test_files:
        print(f"  Parsing {fpath.name}...")
        data = parse_xml_file(fpath)
        stats = compute_data_quality(data)
        stats["split"] = "testing"
        all_stats.append(stats)
        print(f"    Patient {stats['patient_id']}: "
              f"glucose={stats['n_glucose']}, bolus={stats['n_bolus']}, "
              f"meal={stats['n_meal']}, basal={stats['n_basal']}, "
              f"windows={stats['n_windows_30min']}, "
              f"missing={stats['missing_rate']:.2%}, "
              f"usable={stats['usable']}")

    df = pd.DataFrame(all_stats)
    return df


if __name__ == "__main__":
    from configs.default_config import DATA_DIR

    train_dir = DATA_DIR / "raw" / "training"
    test_dir = DATA_DIR / "raw" / "testing"

    if not train_dir.exists():
        # Fallback to OhioT1DM directory
        train_dir = Path(__file__).parent.parent / "OhioT1DM" / "train"
        test_dir = Path(__file__).parent.parent / "OhioT1DM" / "test"

    df = parse_all_patients(train_dir, test_dir)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
