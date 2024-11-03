
## NUMERICAL CONSTANTS ##
####  Time  ####
SECONDS_PER_MINUTE = MINUTES_PER_HOUR = 60
MICROSECONDS_PER_MILLISECOND = MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_SECOND = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND  # 1,000,000

####  Tobii Monitor  ####
TOBII_REFRESH_RATE = 100  # Hz
TOBII_RESOLUTION = (1920, 1080)  # pixels
TOBII_WIDTH, TOBII_HEIGHT = 53.1, 30.0  # cm


## STRING CONSTANTS ##

####  General  ####
ID_STR = "id"
NAME_STR = "name"
TYPE_STR = "type"
COLOR_STR = "color"
URL_STR = "url"
ARTICLES_STR = "articles"
LICENSE_STR = "license"
OUTPUT_STR = "output"
WIDTH_STR, HEIGHT_STR = "width", "height"
RESOLUTION_STR = "resolution"
SAMPLING_RATE_STR = "sampling_rate"
PIXEL_STR = "pixel"
ITERATION_STR = "iteration"
DISTANCE_STR = "distance"
DIFFERENCE_STR = "difference"
THRESHOLD_STR = "threshold"
FIELD_NAME_STR = f"field_{NAME_STR}"
RUNTIME_STR = "runtime"

#### Trial Field Names  ####
SUBJECT_STR = "subject"
SUBJECT_ID_STR = f"{SUBJECT_STR}_{ID_STR}"
TRIAL_STR = "trial"
TRIAL_ID_STR = f"{TRIAL_STR}_{ID_STR}"
STIMULUS_STR = "stimulus"
STIMULUS_TYPE_STR = f"{STIMULUS_STR}_{TYPE_STR}"
STIMULUS_NAME_STR = f"{STIMULUS_STR}_{NAME_STR}"
CHANNEL_STR = "channel"
CHANNEL_TYPE_STR = f"{CHANNEL_STR}_{TYPE_STR}"

# TODO: replace these with enum
IMAGE_STR = "image"
VIDEO_STR = "video"
MOVING_DOT_STR = "moving_dot"

####  Eye Tracking Field Names  ####
TIME_STR = "time"
T, X, Y = "t", "x", "y"
PUPIL = "pupil"
LEFT_X, LEFT_Y, LEFT_PUPIL = f"left_{X}", f"left_{Y}", f"left_{PUPIL}"
RIGHT_X, RIGHT_Y, RIGHT_PUPIL = f"right_{X}", f"right_{Y}", f"right_{PUPIL}"

LABELER_STR = "labeler"
STATUS_STR = "status"
VIEWER_DISTANCE_STR = f"viewer_{DISTANCE_STR}"
PIXEL_SIZE_STR = f"{PIXEL_STR}_size"
IS_OUTLIER_STR = "is_outlier"
METADATA_STR = "metadata"

SAMPLE_STR, SAMPLES_STR = "sample", "samples"
LABEL_STR, LABELS_STR = "label", "labels"
EVENT_STR, EVENTS_STR = "event", "events"
MATCH_STR, MATCHES_STR = "match", "matches"
METRIC_STR, METRICS_STR = "metric", "metrics"

SAMPLES_CHANNEL_STR = f"{SAMPLES_STR}_{CHANNEL_STR}"
EVENTS_CHANNEL_STR = f"{EVENTS_STR}_{CHANNEL_STR}"

####  Event-Features  ####
START_STR, END_STR = "start", "end"
ONSET_STR, OFFSET_STR = "onset", "offset"
FEATURE_STR, FEATURES_STR = "feature", "features"
DURATION_STR = "duration"
VELOCITY_STR = "velocity"
ACCELERATION_STR = "acceleration"
AMPLITUDE_STR = "amplitude"
AZIMUTH_STR = "azimuth"
COUNT_STR = "count"
AREA_STR = "area"
MAX_DURATION_STR, MIN_DURATION_STR = f"max_{DURATION_STR}", f"min_{DURATION_STR}"
START_TIME_STR, END_TIME_STR = f"{START_STR}_{TIME_STR}", f"{END_STR}_{TIME_STR}"
CUMULATIVE_DISTANCE_STR = f"cumulative_{DISTANCE_STR}"
CUMULATIVE_AMPLITUDE_STR = f"cumulative_{AMPLITUDE_STR}"
PEAK_VELOCITY_STR, MEDIAN_VELOCITY_STR = f"peak_{VELOCITY_STR}", f"median_{VELOCITY_STR}"
CENTER_PIXEL_STR = f"center_{PIXEL_STR}"
PIXEL_STD_STR = f"{PIXEL_STR}_std"
DISPERSION_STR = "dispersion"
ELLIPSE_AREA_STR = f"ellipse_{AREA_STR}"

####  Metrics  ####
ACCURACY_STR = "accuracy"
BALANCED_ACCURACY_STR = "balanced_accuracy"
COHENS_KAPPA_STR = "cohen's_kappa"
MCC_STR = "mcc"
COMPLEMENT_NLD_STR = "complement_nld"
PRECISION_STR = "precision"
RECALL_STR = "recall"
F1_STR = "f1"
D_PRIME_STR = "d_prime"
CRITERION_STR = "criterion"
FALSE_ALARM_RATE_STR = "false_alarm_rate"
MATCH_RATIO_STR = f"{MATCH_STR}_ratio"
