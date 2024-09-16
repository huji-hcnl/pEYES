from analysis._default_values.analysis_config import DATASET_NAME, PROCESSED_DATA_DIR
from analysis.process.full_pipeline import run

###################

results = run(output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, verbose=False)
fix_results = run(
    output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, pos_labels=1, verbose=False
)
sac_results = run(
    output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, pos_labels=2, verbose=False
)
