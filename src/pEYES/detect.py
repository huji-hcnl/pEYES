from typing import Union, Tuple, Dict, Any

import numpy as np

from src.pEYES._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, EngbertDetector, NHDetector, REMoDNaVDetector
)


def detect(
        detector_name: str,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
        include_metadata: bool = False,
        **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    detector = _create_detector(detector_name, **kwargs)
    labels, metadata = detector.detect(t, x, y, viewer_distance, pixel_size)
    if include_metadata:
        return labels, metadata
    return labels


def _create_detector(
        detector_name: str,
        **kwargs
):
    detector_name_lower = detector_name.lower().strip().replace('-', '')
    if detector_name_lower == 'ivt':
        default_params = IVTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IVTDetector(**kwargs)
    elif detector_name_lower == 'ivvt':
        default_params = IVVTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IVVTDetector(**kwargs)
    elif detector_name_lower == 'idt':
        default_params = IDTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IDTDetector(**kwargs)
    elif detector_name_lower == 'engbert':
        default_params = EngbertDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return EngbertDetector(**kwargs)
    elif detector_name_lower == 'nh':
        default_params = NHDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return NHDetector(**kwargs)
    elif detector_name_lower == 'remodnav':
        default_params = REMoDNaVDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return REMoDNaVDetector(**kwargs)
    else:
        raise NotImplementedError(f'Detector `{detector_name}` is not implemented.')
