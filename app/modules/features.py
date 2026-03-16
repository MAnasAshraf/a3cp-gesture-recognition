import numpy as np

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = 468
NUM_ANGLES = 14
FEATURE_SIZE = (NUM_POSE + NUM_HAND + NUM_FACE) * 3 + NUM_ANGLES  # = 1580

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
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)
    return keyframes if keyframes else list(range(len(landmarks)))

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_landmarks(results):
    """Extract feature vector from MediaPipe holistic results. Returns np.array of shape (1580,) or None."""
    landmarks = []

    # Pose (33 * 3 = 99)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
    else:
        landmarks.extend([0.0] * NUM_POSE * 3)

    # Left hand (21 * 3 = 63) + angles (14)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
        h = results.left_hand_landmarks.landmark
        angles = [
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
        landmarks.extend(angles)
    else:
        landmarks.extend([0.0] * (NUM_HAND * 3 + NUM_ANGLES))

    # Face (468 * 3 = 1404)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.visibility])
    else:
        landmarks.extend([0.0] * NUM_FACE * 3)

    arr = np.array(landmarks)
    if len(arr) != FEATURE_SIZE:
        return None
    return arr
