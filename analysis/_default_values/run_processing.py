from analysis._default_values._helpers import DATASET_NAME, PROCESSED_DATA_DIR, DEFAULTS_DETECTORS
from analysis.process.full_pipeline import full_pipeline

###################

results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DEFAULTS_DETECTORS.values(),
    verbose=False
)

fix_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DEFAULTS_DETECTORS.values(),
    pos_labels=1,
    verbose=False
)

sac_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DEFAULTS_DETECTORS.values(),
    pos_labels=2,
    verbose=False
)
