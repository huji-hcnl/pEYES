import numpy as np
import plotly.graph_objects as go

from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum


# TODO: FIX THESE FIGURES + ADD DOCSTRINGS + CALCULATE RANSAC REGRESSION LINES


def peak_velocity_to_amplitude(events: EventSequenceType) -> go.Figure:
    saccades = [e for e in events if e.label == EventLabelEnum.SACCADE]
    amps = np.array([s.amplitude for s in saccades])
    peak_vels = np.array([np.nanmax(s.velocities(unit="deg")) for s in saccades])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=amps, y=peak_vels, mode="markers"))
    # TODO: Add regression line
    fig.update_layout(
        title="Peak Velocity vs. Amplitude",
        xaxis_title="Amplitude (deg)",
        yaxis_title="Peak Velocity (deg/s)",
    )
    return fig


def duration_to_amplitude(events: EventSequenceType) -> go.Figure:
    saccades = [e for e in events if e.label == EventLabelEnum.SACCADE]
    amps = np.array([s.amplitude for s in saccades])
    durations = np.array([s.duration for s in saccades])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=amps, y=durations, mode="markers"))
    # TODO: Add regression line
    fig.update_layout(
        title="Duration vs. Amplitude",
        xaxis_title="Amplitude (deg)",
        yaxis_title="Duration (ms)",
    )
    return fig
