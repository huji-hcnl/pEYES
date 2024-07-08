
##  Time  ##
SECONDS_PER_MINUTE = MINUTES_PER_HOUR = 60
MICROSECONDS_PER_MILLISECOND = MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_SECOND = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND  # 1,000,000

##  Tobii Monitor  ##
TOBII_REFRESH_RATE = 100  # Hz
TOBII_RESOLUTION = (1920, 1080)  # pixels
TOBII_WIDTH, TOBII_HEIGHT = 53.1, 30.0  # cm

##  Strings  ##
MINUTE_STR, MINUTES_STR = "minute", "minutes"
SECOND_STR, SECONDS_STR = "second", "seconds"
MILLISECOND_STR, MILLISECONDS_STR = "millisecond", "milliseconds"
MICROSECOND_STR, MICROSECONDS_STR = "microsecond", "microseconds"

SAMPLE_STR, SAMPLES_STR = "sample", "samples"
LABEL_STR, LABELS_STR = "label", "labels"
EVENT_STR, EVENTS_STR = "event", "events"
PIXEL_STR, PIXELS_STR = "pixel", "pixels"
DEGREE_STR, DEGREES_STR = "degree", "degrees"
WIDTH_STR, HEIGHT_STR = "width", "height"
RESOLUTION_STR = "resolution"

ID_STR = "id"
SUBJECT_STR = "subject"
TRIAL_STR = "trial"
STIMULUS_STR = "stimulus"
NAME_STR = "name"
TYPE_STR = "type"
SUBJECT_ID_STR, TRIAL_ID_STR = f"{SUBJECT_STR}_{ID_STR}", f"{TRIAL_STR}_{ID_STR}"
STIMULUS_TYPE_STR, STIMULUS_NAME_STR = f"{STIMULUS_STR}_{TYPE_STR}", f"{STIMULUS_STR}_{NAME_STR}"
IMAGE_STR = "image"
VIDEO_STR = "video"
MOVING_DOT_STR = "moving_dot"
URL_STR = "url"
ARTICLE_STR, ARTICLES_STR = "article", "articles"

T, X, Y = "t", "x", "y"
PUPIL = "pupil"
LEFT_X, LEFT_Y, LEFT_PUPIL = "left_x", "left_y", "left_pupil"
RIGHT_X, RIGHT_Y, RIGHT_PUPIL = "right_x", "right_y", "right_pupil"

VIEWER_DISTANCE_STR = "viewer_distance"
PIXEL_SIZE_STR = f"{PIXEL_STR}_size"
IS_OUTLIER_STR = "is_outlier"
SAMPLING_RATE_STR = "sampling_rate"

STATUS_STR = "status"
TIME_STR = "time"
COLOR_STR = "color"
DURATION_STR = "duration"
DISTANCE_STR = "distance"
VELOCITY_STR = "velocity"
ACCELERATION_STR = "acceleration"
AMPLITUDE_STR = "amplitude"
AZIMUTH_STR = "azimuth"
ANGLE_STR = "angle"
COUNT_STR = "count"

MAX_DURATION_STR, MIN_DURATION_STR = "max_duration", "min_duration"
START_TIME_STR, END_TIME_STR = "start_time", "end_time"
CUMULATIVE_DISTANCE_STR = f"cumulative_{DISTANCE_STR}"
CUMULATIVE_AMPLITUDE_STR = f"cumulative_{AMPLITUDE_STR}"
PEAK_VELOCITY_STR, MEDIAN_VELOCITY_STR = f"peak_{VELOCITY_STR}", f"median_{VELOCITY_STR}"
CENTER_PIXEL_STR = f"center_{PIXELS_STR}"
PIXEL_STD_STR = f"{PIXELS_STR}_std"
