"""
ROI Extractor Module

Extracts facial Regions of Interest (ROI) using dlib face detection
and 68-point facial landmark prediction.

ROI Components:
- mulut (mouth): landmarks 48-68
- mata_kiri (left eye): landmarks 17-22 + 36-42
- mata_kanan (right eye): landmarks 22-27 + 42-48
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import dlib
import numpy as np


class ROIConfig(NamedTuple):
    """Configuration for a single ROI region."""

    indices: list[int]
    target_size: tuple[int, int]  # (width, height)


# Landmark indices for each facial region
REGIONS: dict[str, ROIConfig] = {
    "mata_kiri": ROIConfig(
        indices=list(range(17, 22)) + list(range(36, 42)),
        target_size=(48, 32),
    ),
    "mata_kanan": ROIConfig(
        indices=list(range(22, 27)) + list(range(42, 48)),
        target_size=(48, 32),
    ),
    "mulut": ROIConfig(
        indices=list(range(48, 68)),
        target_size=(70, 35),
    ),
}

# Padding for ROI extraction
PADDING_X = 6
PADDING_Y = 8


class ROIExtractor:
    """
    Facial ROI extractor using dlib.

    Detects faces and extracts specific facial regions (eyes, mouth)
    using 68-point facial landmark prediction.
    """

    def __init__(self, predictor_path: str | Path):
        """
        Initialize ROI extractor.

        Args:
            predictor_path: Path to dlib's shape_predictor_68_face_landmarks.dat file

        Raises:
            FileNotFoundError: If predictor file doesn't exist
        """
        predictor_path = Path(predictor_path)
        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))

    def extract_region(
        self,
        image: np.ndarray,
        landmarks: dlib.full_object_detection,
        config: ROIConfig,
    ) -> np.ndarray | None:
        """
        Extract a single ROI region from the image.

        Args:
            image: Input image (BGR or grayscale)
            landmarks: Detected facial landmarks
            config: ROI configuration with indices and target size

        Returns:
            Resized ROI image, or None if extraction fails
        """
        pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in config.indices]
        xs, ys = zip(*pts)

        # Calculate bounding box with padding
        left = max(0, min(xs) - PADDING_X)
        top = max(0, min(ys) - PADDING_Y)
        right = min(image.shape[1], max(xs) + PADDING_X)
        bottom = min(image.shape[0], max(ys) + PADDING_Y)

        # Extract and validate ROI
        roi = image[top:bottom, left:right]
        if roi.size == 0:
            return None

        # Resize to target size
        roi = cv2.resize(roi, config.target_size)
        return roi

    def extract_rois_from_frame(
        self, frame: np.ndarray
    ) -> dict[str, np.ndarray] | None:
        """
        Extract all ROI regions from a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Dictionary mapping region names to grayscale ROI images,
            or None if no face is detected
        """
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None

        # Use the first detected face
        landmarks = self.predictor(gray, faces[0])

        # Extract each region
        rois = {}
        for region_name, config in REGIONS.items():
            roi = self.extract_region(gray, landmarks, config)
            if roi is None:
                return None  # All regions must be valid
            rois[region_name] = roi

        return rois

    def extract_rois_from_frames(
        self, frames: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]:
        """
        Extract ROIs from multiple frames.

        Args:
            frames: List of frames (BGR format)

        Returns:
            List of ROI dictionaries for frames where face was detected.
            Frames without detected faces are skipped.
        """
        results = []
        for frame in frames:
            rois = self.extract_rois_from_frame(frame)
            if rois is not None:
                results.append(rois)
        return results

    @staticmethod
    def get_region_names() -> list[str]:
        """Get ordered list of region names."""
        return list(REGIONS.keys())

    @staticmethod
    def get_target_size(region_name: str) -> tuple[int, int]:
        """Get target size for a specific region."""
        return REGIONS[region_name].target_size
