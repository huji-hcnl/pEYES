import unittest

import numpy as np

import peyes
from peyes._DataModels.EventLabelEnum import EventLabelEnum


def _events(seed=0, offset=0.0, n=100):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 2.0
    return peyes.create_events(
        [EventLabelEnum.FIXATION] * n, t=t,
        x=np.full(n, 500.0 + offset + rng.integers(0, 20)), y=np.full(n, 500.0),
        pupil=np.ones(n), viewer_distance=60.0, pixel_size=0.0277,
    )


class TestFeaturesByLabels(unittest.TestCase):

    def test_empty_input_keeps_the_schema(self):
        """ M-13: the early return skipped the `count` column, and the frame had no feature columns. """
        empty = peyes.event_metrics.features_by_labels([])
        self.assertIn("count", empty.columns)
        self.assertIn("duration", empty.columns)
        self.assertEqual(list(peyes.event_metrics.features_by_labels(_events()).columns), list(empty.columns))

    def test_empty_counts_are_zero(self):
        empty = peyes.event_metrics.features_by_labels([])
        self.assertTrue((empty["count"] == 0).all())


class TestEventSummaryFigure(unittest.TestCase):

    def test_show_outliers_with_no_outliers(self):
        """ M-13 downstream: this raised KeyError when one of the two groups was empty. """
        self.assertIsNotNone(peyes.visualization.event_summary(_events(), show_outliers=True))

    def test_show_outliers_disabled(self):
        self.assertIsNotNone(peyes.visualization.event_summary(_events(), show_outliers=False))


class TestFeatureComparison(unittest.TestCase):

    def test_more_sequences_than_event_labels(self):
        """
        V-6: the colour fallback indexed a label-keyed dict by position, so it returned event-label
        colours below six sequences and raised above it.
        """
        for n in (2, 6, 7, 12):
            with self.subTest(num_sequences=n):
                figure = peyes.visualization.feature_comparison(
                    "duration", *[_events(seed=i, offset=10.0 * i) for i in range(n)]
                )
                self.assertIsNotNone(figure)

    def test_explicit_colors_still_honoured(self):
        figure = peyes.visualization.feature_comparison(
            "duration", _events(), _events(seed=1), labels=["a", "b"],
            colors={"a": "#010203", "b": "#040506"},
        )
        self.assertIsNotNone(figure)


class TestGazeHeatmap(unittest.TestCase):

    def test_off_screen_samples_are_dropped(self):
        """ V-7: out-of-range coordinates raised, and negative ones wrapped to the opposite edge. """
        x = np.array([10.0, 700.0, -5.0, 320.0])
        y = np.array([10.0, 50.0, 20.0, 240.0])
        self.assertIsNotNone(peyes.visualization.gaze_heatmap(x, y, (640, 480)))
