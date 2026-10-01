"""
MediaPipe Holistic landmark extraction utilities for KSL.
"""

import numpy as np
import mediapipe as mp
import cv2
from typing import Optional, Tuple, List


# MediaPipe Holistic setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_landmarks(results) -> np.ndarray:
    """
    Extract and flatten all landmarks from MediaPipe Holistic results.
    Returns a fixed-size vector of shape (1662,).
    Missing parts are filled with zeros.
    """
    pose = np.zeros(33 * 4)       # x, y, z, visibility
    face = np.zeros(468 * 3)      # x, y, z
    left_hand = np.zeros(21 * 3)  # x, y, z
    right_hand = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()

    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        ).flatten()

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    return np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)


def draw_landmarks(image: np.ndarray, results) -> np.ndarray:
    """Draw pose, face, and hand landmarks on the image."""
    # Pose
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    # Face (simplified)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    # Hands
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    return image


class LandmarkExtractor:
    """Context manager for MediaPipe Holistic."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.holistic = None

    def __enter__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.holistic:
            self.holistic.close()

    def process(self, image_rgb: np.ndarray):
        """Process an RGB image and return results."""
        return self.holistic.process(image_rgb)

    def get_landmarks(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Full pipeline: BGR image → RGB → process → landmarks + results.
        Returns (landmarks_vector, results)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.process(image_rgb)
        image_rgb.flags.writeable = True
        landmarks = extract_landmarks(results)
        return landmarks, results
