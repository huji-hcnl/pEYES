import time

from analysis._article_results.hfc._helpers import DATASET_NAME, PROCESSED_DATA_DIR, DETECTORS, MATCHING_SCHEMES
from analysis.process.full_pipeline import full_pipeline

###################

start = time.time()
print(f"Processing '{DATASET_NAME}' dataset with {len(DETECTORS)} detectors")

print("(1) GLOBAL pipeline")
results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    matching_schemes=MATCHING_SCHEMES,
    num_iterations=1,
    verbose=False
)
end = time.time()
print(f"\tCompleted GLOBAL pipeline in {end - start:.2f} seconds")

print("(2) FIXATION and SACCADE pipelines")
fix_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    matching_schemes=MATCHING_SCHEMES,
    pos_labels=1,
    num_iterations=1,
    verbose=False
)
print(f"\tCompleted FIXATION pipeline in {time.time() - end:.2f} seconds")
end = time.time()

print(f"Completed full pipeline in {end - start:.2f} seconds")
