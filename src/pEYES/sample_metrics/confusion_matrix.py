from typing import Sequence, Optional

import pandas as pd
import sklearn.metrics as met

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

_GROUND_TRUTH_STR = "Ground Truth"
_PREDICTION_STR = "Prediction"


def confusion_matrix(
        ground_truth: Sequence[EventLabelEnum],
        prediction: Sequence[EventLabelEnum],
        labels: Optional[Sequence[EventLabelEnum]] = None,
) -> pd.DataFrame:
    labels = list(set(EventLabelEnum)) if labels is None else list(set(labels))
    conf = met.confusion_matrix(ground_truth, prediction, labels=labels)
    df = pd.DataFrame(conf, index=labels, columns=labels)
    df.index.name = _GROUND_TRUTH_STR
    df.columns.name = _PREDICTION_STR
    return df

