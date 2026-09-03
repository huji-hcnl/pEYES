import unittest

import numpy as np
import plotly.graph_objects as go

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

    @staticmethod
    def _clustered_gaze(seed=0, n=500):
        rng = np.random.default_rng(seed)
        x = np.concatenate([rng.normal(60, 15, n * 3 // 5), rng.normal(140, 10, n * 2 // 5)])
        y = np.concatenate([rng.normal(60, 15, n * 3 // 5), rng.normal(140, 10, n * 2 // 5)])
        return x, y

    def test_scale_matching_sigma_squared_matches_default(self):
        """
        V-13: `scale` used to have no effect at all (cancelled exactly by the min-max normalization that
        always follows it). Passing exactly sigma**2 - what every pre-existing caller in analysis/ does -
        must still reproduce the un-scaled default exactly.
        """
        x, y = self._clustered_gaze()
        explicit = peyes.visualization.gaze_heatmap(x, y, (200, 200), sigma=10, scale=100)
        default = peyes.visualization.gaze_heatmap(x, y, (200, 200), sigma=10)
        np.testing.assert_array_equal(
            np.array(explicit.data[1].z, dtype=float), np.array(default.data[1].z, dtype=float),
        )

    def test_scale_now_has_a_real_monotonic_effect(self):
        """ V-13: a larger scale must now reveal at least as much of the heatmap as a smaller one. """
        x, y = self._clustered_gaze()
        visible_counts = []
        for scale in (10, 100, 1000):
            fig = peyes.visualization.gaze_heatmap(x, y, (200, 200), sigma=10, scale=scale)
            z = np.array(fig.data[1].z, dtype=float)
            visible_counts.append(np.sum(~np.isnan(z)))
        self.assertLess(visible_counts[0], visible_counts[1])
        self.assertLessEqual(visible_counts[1], visible_counts[2])

    def test_rejects_non_positive_scale(self):
        x, y = self._clustered_gaze()
        self.assertRaises(ValueError, peyes.visualization.gaze_heatmap, x, y, (200, 200), scale=0)
        self.assertRaises(ValueError, peyes.visualization.gaze_heatmap, x, y, (200, 200), scale=-1)


class TestScarfplot(unittest.TestCase):
    """
    V-4/V-5: colored band placement used to be ranked by position *among only the labels present in this row*,
    while the heatmap's zmin/zmax always normalized against the full fixed 0-5 EventLabelEnum range - the two
    only agreed when all 6 labels happened to be present. A concrete break: with only PSO(3) and
    SMOOTH_PURSUIT(4) present, the old code rendered PSO in SMOOTH_PURSUIT's color.
    """

    @staticmethod
    def _rendered_color(fig, label: EventLabelEnum) -> str:
        hm = fig.data[0]
        zmin, zmax = hm.zmin, hm.zmax
        norm = (label - zmin) / (zmax - zmin)
        for i in range(0, len(hm.colorscale) - 1, 2):
            lo, hi = hm.colorscale[i][0], hm.colorscale[i + 1][0]
            if lo - 1e-9 <= norm <= hi + 1e-9:
                return hm.colorscale[i][1]
        raise AssertionError(f"{label} matched no colorscale band")

    def _expected_color(self, label: EventLabelEnum) -> str:
        from peyes._utils.visualization_utils import get_label_colormap, to_rgb
        return f"rgb{to_rgb(get_label_colormap(None)[label])}"

    def test_all_six_labels_present_matches_defaults(self):
        t = np.arange(6)
        labels = list(EventLabelEnum)
        fig = peyes.visualization.add_scarfplot_to_figure(go.Figure(), t, labels, top=1, bottom=0)
        for label in EventLabelEnum:
            self.assertEqual(self._expected_color(label), self._rendered_color(fig, label))

    def test_two_non_adjacent_labels_use_their_own_colors(self):
        """ The concrete regression case: PSO+SMOOTH_PURSUIT used to collide onto one color. """
        t = np.arange(6)
        labels = [EventLabelEnum.PSO] * 3 + [EventLabelEnum.SMOOTH_PURSUIT] * 3
        fig = peyes.visualization.add_scarfplot_to_figure(go.Figure(), t, labels, top=1, bottom=0)
        pso_color = self._rendered_color(fig, EventLabelEnum.PSO)
        sp_color = self._rendered_color(fig, EventLabelEnum.SMOOTH_PURSUIT)
        self.assertEqual(self._expected_color(EventLabelEnum.PSO), pso_color)
        self.assertEqual(self._expected_color(EventLabelEnum.SMOOTH_PURSUIT), sp_color)
        self.assertNotEqual(pso_color, sp_color)

    def test_colorbar_ticks_only_list_present_labels(self):
        t = np.arange(6)
        labels = [EventLabelEnum.PSO] * 3 + [EventLabelEnum.SMOOTH_PURSUIT] * 3
        fig = peyes.visualization.add_scarfplot_to_figure(go.Figure(), t, labels, top=1, bottom=0)
        self.assertEqual(("PSO", "SMOOTH_PURSUIT"), fig.data[0].colorbar.ticktext)
