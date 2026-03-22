import numpy as np

NUM_POSE   = 33
NUM_HAND   = 21
NUM_FACE   = 468
NUM_ANGLES = 14  # angles per hand (left and right)
# Layout: pose(99) + left_hand(63) + left_angles(14) + right_hand(63) + right_angles(14) + face(1404)
FEATURE_SIZE = (NUM_POSE + NUM_HAND + NUM_HAND + NUM_FACE) * 3 + NUM_ANGLES * 2  # = 1657

# Column index boundaries (used by face trainer/recognizer)
FACE_COL_START = NUM_POSE * 3 + NUM_HAND * 3 + NUM_ANGLES + NUM_HAND * 3 + NUM_ANGLES  # = 253
FACE_COL_END   = FEATURE_SIZE  # = 1657

# Delta features: body columns (pose+hands) only
DELTA_FEATURE_SIZE = FACE_COL_START    # = 253
FULL_FEATURE_SIZE  = FEATURE_SIZE + DELTA_FEATURE_SIZE  # = 1910

# Shoulder landmark indices in pose block (used for centroid normalization)
_SHOULDER_L_IDX = 11  # MediaPipe pose landmark 11 = left shoulder
_SHOULDER_R_IDX = 12  # MediaPipe pose landmark 12 = right shoulder

# Cache for last known mid-shoulder position (fallback when pose is lost)
_last_mid_shoulder = None

# Angle column indices (used for normalisation during training/inference)
LEFT_ANGLE_COLS  = np.arange(NUM_POSE * 3 + NUM_HAND * 3,
                              NUM_POSE * 3 + NUM_HAND * 3 + NUM_ANGLES)           # 162–175
RIGHT_ANGLE_COLS = np.arange(NUM_POSE * 3 + NUM_HAND * 3 + NUM_ANGLES + NUM_HAND * 3,
                              NUM_POSE * 3 + NUM_HAND * 3 + NUM_ANGLES + NUM_HAND * 3 + NUM_ANGLES)  # 239–252
ALL_ANGLE_COLS   = np.concatenate([LEFT_ANGLE_COLS, RIGHT_ANGLE_COLS])


def calculate_velocity(landmarks):
    velocities = []
    for i in range(1, len(landmarks)):
        velocity = np.linalg.norm(landmarks[i] - landmarks[i-1])
        velocities.append(velocity)
    return np.array(velocities)

def calculate_acceleration(velocities):
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = np.abs(velocities[i] - velocities[i-1])
        accelerations.append(acceleration)
    return np.array(accelerations)

def identify_keyframes(landmarks, velocity_threshold=0.1, acceleration_threshold=0.1):
    if len(landmarks) < 3:
        return list(range(len(landmarks)))
    # Use only pose + hand columns (0–252) for velocity/acceleration.
    # Face landmarks (253–1656) dominate the L2 norm and dilute hand movement,
    # causing keyframes to be missed for hand-dominant gestures like clapping.
    body_cols = landmarks[:, :FACE_COL_START] if landmarks.ndim == 2 else landmarks
    velocities = calculate_velocity(body_cols)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)
    return keyframes if keyframes else list(range(len(landmarks)))

def compute_deltas(sequence: np.ndarray) -> np.ndarray:
    """Append body-column deltas (velocity) to each frame.

    Input:  (T, 1657)  — centroid-normalized frames
    Output: (T, 1910)  — base features + frame-to-frame deltas for cols 0–252
    """
    body = sequence[:, :FACE_COL_START]  # (T, 253)
    deltas = np.zeros_like(body)
    deltas[1:] = body[1:] - body[:-1]   # first frame delta = 0
    return np.concatenate([sequence, deltas], axis=1)


def _centroid_normalize(arr: np.ndarray) -> np.ndarray:
    """Subtract mid-shoulder (x,y) from all landmark coordinates in-place.

    Skips angle columns (162–175, 239–252) and visibility values.
    Uses cached mid-shoulder if pose is missing for the current frame.
    """
    global _last_mid_shoulder

    # Mid-shoulder from pose landmarks 11 and 12 (each is x,y,vis triplet)
    lx = arr[_SHOULDER_L_IDX * 3]       # pose[11].x
    ly = arr[_SHOULDER_L_IDX * 3 + 1]   # pose[11].y
    rx = arr[_SHOULDER_R_IDX * 3]       # pose[12].x
    ry = arr[_SHOULDER_R_IDX * 3 + 1]   # pose[12].y

    has_pose = not (lx == 0.0 and ly == 0.0 and rx == 0.0 and ry == 0.0)

    if has_pose:
        mid_x = (lx + rx) / 2.0
        mid_y = (ly + ry) / 2.0
        _last_mid_shoulder = (mid_x, mid_y)
    elif _last_mid_shoulder is not None:
        mid_x, mid_y = _last_mid_shoulder
    else:
        return arr  # no shoulder reference ever — skip normalization

    # Ranges of (x, y, vis) triplets to normalize (skip angle blocks)
    triplet_ranges = [
        (0, NUM_POSE * 3),          # pose:       0–98
        (99, 99 + NUM_HAND * 3),    # left hand:  99–161
        # 162–175 = left angles → SKIP
        (176, 176 + NUM_HAND * 3),  # right hand: 176–238
        # 239–252 = right angles → SKIP
        (FACE_COL_START, FACE_COL_END),  # face: 253–1656
    ]
    for start, end in triplet_ranges:
        arr[start:end:3] -= mid_x    # x values
        arr[start + 1:end:3] -= mid_y  # y values
        # visibility (every 3rd+2) unchanged

    return arr


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def _hand_angles(h):
    """Calculate 14 joint angles from a hand landmark list."""
    return [
        calculate_angle([h[1].x,h[1].y],[h[2].x,h[2].y],[h[3].x,h[3].y]),
        calculate_angle([h[2].x,h[2].y],[h[3].x,h[3].y],[h[4].x,h[4].y]),
        calculate_angle([h[0].x,h[0].y],[h[5].x,h[5].y],[h[6].x,h[6].y]),
        calculate_angle([h[5].x,h[5].y],[h[6].x,h[6].y],[h[7].x,h[7].y]),
        calculate_angle([h[6].x,h[6].y],[h[7].x,h[7].y],[h[8].x,h[8].y]),
        calculate_angle([h[0].x,h[0].y],[h[9].x,h[9].y],[h[10].x,h[10].y]),
        calculate_angle([h[9].x,h[9].y],[h[10].x,h[10].y],[h[11].x,h[11].y]),
        calculate_angle([h[10].x,h[10].y],[h[11].x,h[11].y],[h[12].x,h[12].y]),
        calculate_angle([h[0].x,h[0].y],[h[13].x,h[13].y],[h[14].x,h[14].y]),
        calculate_angle([h[13].x,h[13].y],[h[14].x,h[14].y],[h[15].x,h[15].y]),
        calculate_angle([h[14].x,h[14].y],[h[15].x,h[15].y],[h[16].x,h[16].y]),
        calculate_angle([h[0].x,h[0].y],[h[17].x,h[17].y],[h[18].x,h[18].y]),
        calculate_angle([h[17].x,h[17].y],[h[18].x,h[18].y],[h[19].x,h[19].y]),
        calculate_angle([h[18].x,h[18].y],[h[19].x,h[19].y],[h[20].x,h[20].y]),
    ]

def extract_landmarks(results):
    """Extract feature vector from MediaPipe holistic results. Returns np.array of shape (1657,) or None."""
    landmarks = []

    # Pose (33 * 3 = 99) — cols 0–98
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
    else:
        landmarks.extend([0.0] * NUM_POSE * 3)

    # Left hand (21 * 3 = 63) + angles (14) — cols 99–175
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
        landmarks.extend(_hand_angles(results.left_hand_landmarks.landmark))
    else:
        landmarks.extend([0.0] * (NUM_HAND * 3 + NUM_ANGLES))

    # Right hand (21 * 3 = 63) + angles (14) — cols 176–252
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
        landmarks.extend(_hand_angles(results.right_hand_landmarks.landmark))
    else:
        landmarks.extend([0.0] * (NUM_HAND * 3 + NUM_ANGLES))

    # Face (468 * 3 = 1404) — cols 253–1656
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
    else:
        landmarks.extend([0.0] * NUM_FACE * 3)

    arr = np.array(landmarks)
    if len(arr) != FEATURE_SIZE:
        return None
    arr = _centroid_normalize(arr)
    return arr
