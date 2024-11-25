import time

from analysis._article_results.lund2013._helpers import DATASET_NAME, PROCESSED_DATA_DIR, DETECTORS
from analysis.process.full_pipeline import full_pipeline

###################

start = time.time()
print(f"Processing '{DATASET_NAME}' dataset with {len(DETECTORS)} detectors")

print("(1) GLOBAL pipeline")
results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
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
    num_iterations=1,
    pos_labels=1,
    verbose=False
)
print(f"\tCompleted FIXATION pipeline in {time.time() - end:.2f} seconds")
end = time.time()

print("(3) SACCADE pipeline")
sac_results = full_pipeline(
    output_dir=PROCESSED_DATA_DIR,
    dataset_name=DATASET_NAME,
    detectors=DETECTORS.values(),
    num_iterations=1,
    pos_labels=2,
    verbose=False
)
print(f"\tCompleted SACCADE pipeline in {time.time() - end:.2f} seconds")
end = time.time()

print(f"Completed full pipeline in {end - start:.2f} seconds")
