from analysis._parameter_exploration._helpers import DATASET_NAME, PROCESSED_DATA_DIR, DETECTORS
from analysis.process.full_pipeline import full_pipeline

###################

results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    num_iterations=1,
    verbose=False
)

fix_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    num_iterations=1,
    pos_labels=1,
    verbose=False
)

sac_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    num_iterations=1,
    pos_labels=2,
    verbose=False
)
