from typing import List, Dict, Any


def suggest_from_sample(sample: dict) -> List[Dict[str, Any]]:
    options = [{"type": "anomaly"}]
    options.extend([
        {"type": "forecasting", "requires": ["time_col", "target"], "suggested_time_col": "date",
         "suggested_target": "value"},
        {"type": "classification", "target": "target"},
        {"type": "clustering", "k_suggestion": 5},
    ])
    return options
