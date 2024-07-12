
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

####  Time  ####
MINUTE_STR, MINUTES_STR = "minute", "minutes"
SECOND_STR, SECONDS_STR = "second", "seconds"
MILLISECOND_STR, MILLISECONDS_STR = "millisecond", "milliseconds"
MICROSECOND_STR, MICROSECONDS_STR = "microsecond", "microseconds"

#### Measurements  ####
WIDTH_STR, HEIGHT_STR = "width", "height"
RESOLUTION_STR = "resolution"
PIXEL_STR, PIXELS_STR = "pixel", "pixels"
DEGREE_STR, DEGREES_STR = "degree", "degrees"
DURATION_STR, DURATIONS_STR = "duration", "durations"
DISTANCE_STR, DISTANCES_STR = "distance", "distances"
VELOCITY_STR, VELOCITIES_STR = "velocity", "velocities"
ACCELERATION_STR, ACCELERATIONS_STR = "acceleration", "accelerations"
AMPLITUDE_STR, AMPLITUDES_STR = "amplitude", "amplitudes"
AZIMUTH_STR, AZIMUTHS_STR = "azimuth", "azimuths"
ANGLE_STR, ANGLES_STR = "angle", "angles"
COUNT_STR, COUNTS_STR = "count", "counts"

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

SAMPLING_RATE_STR = "sampling_rate"
STATUS_STR = "status"
VIEWER_DISTANCE_STR = "viewer_distance"
PIXEL_SIZE_STR = f"{PIXEL_STR}_size"
IS_OUTLIER_STR = "is_outlier"

SAMPLE_STR, SAMPLES_STR = "sample", "samples"
LABEL_STR, LABELS_STR = "label", "labels"
EVENT_STR, EVENTS_STR = "event", "events"

#### Event Field Names  ####
MAX_DURATION_STR, MIN_DURATION_STR = "max_duration", "min_duration"
START_TIME_STR, END_TIME_STR = "start_time", "end_time"
CUMULATIVE_DISTANCE_STR = f"cumulative_{DISTANCE_STR}"
CUMULATIVE_AMPLITUDE_STR = f"cumulative_{AMPLITUDE_STR}"
PEAK_VELOCITY_STR, MEDIAN_VELOCITY_STR = f"peak_{VELOCITY_STR}", f"median_{VELOCITY_STR}"
CENTER_PIXEL_STR = f"center_{PIXEL_STR}"
PIXEL_STD_STR = f"{PIXEL_STR}_std"
ELLIPSE_AREA_STR = "ellipse_area"
