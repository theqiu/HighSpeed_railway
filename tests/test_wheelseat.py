"""Unit tests for wheel seat average parsing and grouping."""

from __future__ import annotations

import unittest

import pandas as pd

from app.analytics import (
    _parse_wheelseat_average,
    wheelseat_average_panels,
    wheelseat_monotonic_analysis,
)


class WheelseatAverageParsingTests(unittest.TestCase):
    """Validate that wheel seat average labels are parsed correctly."""

    def test_parse_common_variants(self) -> None:
        cases = {
            "左轴承内径截面A平均值": ("左轴承内径", "A"),
            "左轴承内径截面 b 平均值": ("左轴承内径", "B"),
            "左轴承内径截面C平均值-1": ("左轴承内径", "C"),
            "左轴承内径截面D平均": ("左轴承内径", "D"),
            "左轴承内径A截面平均值": ("左轴承内径", "A"),
            "左轮座B截面": ("左轮座", "B"),
            "右轴承内径截面a平均值（磨后）": ("右轴承内径", "A"),
            "右轴承内径A平均值": ("右轴承内径", "A"),
        }

        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(_parse_wheelseat_average(label), expected)


class WheelseatAveragePanelTests(unittest.TestCase):
    """Ensure grouped panels combine multiple sections."""

    def test_sections_combined_into_single_panel(self) -> None:
        records = [
            {"qcitem": "左轴承内径截面A平均值", "numeric_value": 10.1, "qc_timestamp": pd.Timestamp("2024-01-01")},
            {"qcitem": "左轴承内径截面B平均值", "numeric_value": 10.2, "qc_timestamp": pd.Timestamp("2024-01-02")},
            {"qcitem": "左轴承内径截面C平均值", "numeric_value": 10.3, "qc_timestamp": pd.Timestamp("2024-01-03")},
            {"qcitem": "左轴承内径截面D平均值", "numeric_value": 10.4, "qc_timestamp": pd.Timestamp("2024-01-04")},
            {"qcitem": "左轮座A截面平均值", "numeric_value": 11.1, "qc_timestamp": pd.Timestamp("2024-01-01")},
            {"qcitem": "左轮座B截面平均值", "numeric_value": 11.2, "qc_timestamp": pd.Timestamp("2024-01-02")},
            {"qcitem": "左轮座C截面平均值", "numeric_value": 11.3, "qc_timestamp": pd.Timestamp("2024-01-03")},
        ]
        sequence = pd.DataFrame.from_records(records)

        panels = wheelseat_average_panels(sequence)

        bearing_panel = panels[panels["panel_name"] == "左轴承内径"]
        self.assertEqual(bearing_panel["section"].tolist(), ["A", "B", "C", "D"])

        seat_panel = panels[panels["panel_name"] == "左轮座"]
        self.assertEqual(seat_panel["section"].tolist(), ["A", "B", "C"])


class WheelseatMonotonicAnalysisTests(unittest.TestCase):
    """Validate the monotonic wheel seat analyser."""

    def _build_dataframe(self, right_b_increase: bool = False) -> pd.DataFrame:
        base_time = pd.Timestamp("2024-01-01")
        entries = []
        left_values = {"A": 12.5, "B": 12.3, "C": 12.1}
        right_values = {"A": 12.4, "B": 12.2, "C": 12.0}
        if right_b_increase:
            right_values["B"] = 12.45

        for idx, (section, value) in enumerate(left_values.items(), start=1):
            entries.append(
                {
                    "xb005": "W1",
                    "qcitem": f"左轮座截面{section}平均值",
                    "inputvalue": str(value),
                    "qc_timestamp": base_time + pd.Timedelta(days=idx),
                    "qcorder": idx,
                }
            )

        for idx, (section, value) in enumerate(right_values.items(), start=1):
            entries.append(
                {
                    "xb005": "W1",
                    "qcitem": f"右轮座截面{section}平均值",
                    "inputvalue": str(value),
                    "qc_timestamp": base_time + pd.Timedelta(days=idx + 5),
                    "qcorder": idx + 10,
                }
            )

        return pd.DataFrame(entries)

    def test_monotonic_summary_passes_when_values_descend(self) -> None:
        df = self._build_dataframe()
        summary, sequences = wheelseat_monotonic_analysis(df)

        self.assertEqual(set(summary["panel_group"]), {"左侧", "右侧"})
        self.assertTrue((summary["violation_count"] == 0).all())
        self.assertTrue(summary["monotonic_descending"].all())
        self.assertIn("panel_name", sequences.columns)
        self.assertTrue(all("平均" in item for item in sequences["qcitem"].dropna().unique()))

    def test_monotonic_summary_flags_increasing_section(self) -> None:
        df = self._build_dataframe(right_b_increase=True)
        summary, _ = wheelseat_monotonic_analysis(df)

        self.assertFalse(summary.empty)
        right_side = summary[summary["panel_group"] == "右侧"].iloc[0]
        self.assertGreater(int(right_side["violation_count"]), 0)
        self.assertFalse(bool(right_side["monotonic_descending"]))
        self.assertIn("右轮座", str(right_side.get("violating_panels", "")))


if __name__ == "__main__":
    unittest.main()
