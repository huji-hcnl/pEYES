
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
ARTICLE_STR, ARTICLES_STR = "article", "articles"
START_STR, END_STR = "start", "end"
ONSET_STR, OFFSET_STR = "onset", "offset"

####  Time  ####
MINUTE_STR, MINUTES_STR = "minute", "minutes"
SECOND_STR, SECONDS_STR = "second", "seconds"
MILLISECOND_STR, MILLISECONDS_STR = "millisecond", "milliseconds"
MICROSECOND_STR, MICROSECONDS_STR = "microsecond", "microseconds"

####  Measurements & Units  ####
WIDTH_STR, HEIGHT_STR = "width", "height"
RESOLUTION_STR = "resolution"
SAMPLING_RATE_STR = "sampling_rate"
PIXEL_STR, PIXELS_STR = "pixel", "pixels"
DEGREE_STR, DEGREES_STR = "degree", "degrees"
ANGLE_STR, ANGLES_STR = "angle", "angles"
ITERATION_STR, ITERATIONS_STR = "iteration", "iterations"

#### Trial Field Names  ####
SUBJECT_STR = "subject"
SUBJECT_ID_STR = f"{SUBJECT_STR}_{ID_STR}"
TRIAL_STR = "trial"
TRIAL_ID_STR = f"{TRIAL_STR}_{ID_STR}"
STIMULUS_STR = "stimulus"
STIMULUS_TYPE_STR = f"{STIMULUS_STR}_{TYPE_STR}"
STIMULUS_NAME_STR = f"{STIMULUS_STR}_{NAME_STR}"

IMAGE_STR = "image"
VIDEO_STR = "video"
MOVING_DOT_STR = "moving_dot"

####  Eye Tracking Field Names  ####
TIME_STR = "time"
T, X, Y = "t", "x", "y"
PUPIL = "pupil"
LEFT_X, LEFT_Y, LEFT_PUPIL = f"left_{X}", f"left_{Y}", f"left_{PUPIL}"
RIGHT_X, RIGHT_Y, RIGHT_PUPIL = f"right_{X}", f"right_{Y}", f"right_{PUPIL}"

STATUS_STR = "status"
VIEWER_DISTANCE_STR = "viewer_distance"
PIXEL_SIZE_STR = f"{PIXEL_STR}_size"
IS_OUTLIER_STR = "is_outlier"

SAMPLE_STR, SAMPLES_STR = "sample", "samples"
LABEL_STR, LABELS_STR = "label", "labels"
EVENT_STR, EVENTS_STR = "event", "events"

####  Event-Features  ####
FEATURE_STR, FEATURES_STR = "feature", "features"
DURATION_STR, DURATIONS_STR = "duration", "durations"
VELOCITY_STR, VELOCITIES_STR = "velocity", "velocities"
ACCELERATION_STR, ACCELERATIONS_STR = "acceleration", "accelerations"
DISTANCE_STR, DISTANCES_STR = "distance", "distances"
DIFFERENCE_STR, DIFFERENCES_STR = "difference", "differences"
AMPLITUDE_STR, AMPLITUDES_STR = "amplitude", "amplitudes"
AZIMUTH_STR, AZIMUTHS_STR = "azimuth", "azimuths"
COUNT_STR, COUNTS_STR = "count", "counts"
AREA_STR = "area"
THRESHOLD_STR = "threshold"
MAX_DURATION_STR, MIN_DURATION_STR = "max_duration", "min_duration"
START_TIME_STR, END_TIME_STR = f"{START_STR}_{TIME_STR}", f"{END_STR}_{TIME_STR}"
CUMULATIVE_DISTANCE_STR = f"cumulative_{DISTANCE_STR}"
CUMULATIVE_AMPLITUDE_STR = f"cumulative_{AMPLITUDE_STR}"
PEAK_VELOCITY_STR, MEDIAN_VELOCITY_STR = f"peak_{VELOCITY_STR}", f"median_{VELOCITY_STR}"
CENTER_PIXEL_STR = f"center_{PIXEL_STR}"
PIXEL_STD_STR = f"{PIXEL_STR}_std"
ELLIPSE_AREA_STR = f"ellipse_{AREA_STR}"

####  Metrics  ####
METRIC_STR, METRICS_STR = "metric", "metrics"
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
