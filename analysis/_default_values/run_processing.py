from analysis._default_values.analysis_config import DATASET_NAME, PROCESSED_DATA_DIR
from analysis.process.full_pipeline import full_pipeline

###################

results = full_pipeline(output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, verbose=False)
fix_results = full_pipeline(output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, pos_labels=1, verbose=False)
sac_results = full_pipeline(output_dir=PROCESSED_DATA_DIR, dataset_name=DATASET_NAME, pos_labels=2, verbose=False)
