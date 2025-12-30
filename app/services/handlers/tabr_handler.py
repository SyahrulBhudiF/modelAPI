"""
TabR Model Handler

Handles video preprocessing and TabR model inference.
Pipeline: Video → Frames → ROI (dlib) → POC-ABS → Feature Vector → TabR
"""

import tempfile
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from app.models.tabr import LitTabR
from app.preprocessing import POC, Quadran, ROIExtractor, Vektor
from app.services.base import BaseHandler

# Component processing order (must match training)
COMPONENTS = ["mata_kiri", "mata_kanan", "mulut"]

# Block size for POC computation
BLOCK_SIZE = 7


class TabRHandler(BaseHandler):
    """
    Handler for TabR model inference.

    Responsible for:
    - Loading TabR checkpoint and preprocessing artifacts
    - Video preprocessing (ROI extraction via dlib, POC-ABS features)
    - Feature extraction and inference
    """

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.model: LitTabR | None = None
        self.imputer = None
        self.scaler = None
        self.feature_cols: list[str] | None = None
        self.context_x: torch.Tensor | None = None
        self.context_y: torch.Tensor | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ROI extractor (loaded on load())
        self.roi_extractor: ROIExtractor | None = None

    def load(self) -> None:
        """Load TabR model checkpoint, preprocessing artifacts, and dlib predictor."""
        model_path = Path(self.model_dir)

        # Load dlib shape predictor
        predictor_path = model_path / "shape_predictor_68_face_landmarks.dat"
        if not predictor_path.exists():
            raise FileNotFoundError(
                f"dlib shape predictor not found: {predictor_path}. "
                "Please download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        self.roi_extractor = ROIExtractor(predictor_path)

        # Load model checkpoint
        checkpoint_path = model_path / "tabr_model.ckpt"
        if not checkpoint_path.exists():
            # Try alternative naming
            checkpoint_path = model_path / "model.ckpt"
        if not checkpoint_path.exists():
            ckpt_files = list(model_path.glob("*.ckpt"))
            if ckpt_files:
                checkpoint_path = ckpt_files[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")

        self.model = LitTabR.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        # Load preprocessing artifacts
        imputer_path = model_path / "imputer.joblib"
        scaler_path = model_path / "scaler.joblib"
        feature_cols_path = model_path / "feature_cols.joblib"

        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer not found: {imputer_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not feature_cols_path.exists():
            raise FileNotFoundError(f"Feature columns not found: {feature_cols_path}")

        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        feature_cols: list[str] = joblib.load(feature_cols_path)

        self.imputer = imputer
        self.scaler = scaler
        self.feature_cols = feature_cols

        # Load context for TabR retrieval
        context_path = model_path / "context.joblib"
        context_pt_path = model_path / "context.pt"

        if context_path.exists():
            context_x, context_y = joblib.load(context_path)
        elif context_pt_path.exists():
            context_data = torch.load(context_pt_path, map_location=self.device)
            context_x = context_data["ctx_x"]
            context_y = context_data["ctx_y"]
        else:
            raise FileNotFoundError(
                f"Context not found: {context_path} or {context_pt_path}"
            )

        ctx_x = context_x.to(self.device)
        ctx_y = context_y.to(self.device)

        self.context_x = ctx_x
        self.context_y = ctx_y

        # Set up model with preprocessor and context
        self.model.set_preprocessor(imputer, scaler, feature_cols)
        self.model.set_default_context(ctx_x, ctx_y)

        self._is_loaded = True

    def preprocess(self, input_data: list[bytes] | bytes | Any) -> pd.DataFrame:
        """
        Preprocess video input into feature DataFrame.

        Pipeline: Video Segments → Frames → ROI (dlib) → POC → Vektor → Quadran
                  → Flatten Features

        Args:
            input_data: Can be:
                - list[bytes]: Multiple video segments to be merged into one
                - bytes: Single video file bytes

        Returns:
            DataFrame with POC-ABS features (one row per frame pair).
        """
        # Extract frames from video(s)
        if isinstance(input_data, list):
            all_frames = []
            for segment in input_data:
                segment_frames = self._extract_frames(segment)
                all_frames.extend(segment_frames)
            frames = all_frames
        else:
            frames = self._extract_frames(input_data)

        if len(frames) < 2:
            raise ValueError("Video must have at least 2 frames for POC computation")

        # Extract ROIs for all frames
        if self.roi_extractor is None:
            raise RuntimeError("ROI extractor not loaded. Call load() first.")
        rois_per_frame = self.roi_extractor.extract_rois_from_frames(frames)

        if len(rois_per_frame) < 2:
            raise ValueError(
                "Could not detect face in enough frames. "
                f"Only {len(rois_per_frame)} frames had valid face detections."
            )

        # Compute POC-ABS features for consecutive frame pairs
        feature_rows = []

        # Use first frame as baseline for each component
        baseline = rois_per_frame[0]

        for idx in range(1, len(rois_per_frame)):
            current_rois = rois_per_frame[idx]
            frame_no = idx + 1

            row: dict[str, int | float] = {"frame": frame_no}

            for comp in COMPONENTS:
                baseline_roi = baseline[comp]
                current_roi = current_rois[comp]

                # Compute POC
                poc = POC(baseline_roi, current_roi, BLOCK_SIZE)
                poc_output = poc.get_poc()

                # Extract vectors
                vektor = Vektor(poc_output, BLOCK_SIZE)
                vektor_output = vektor.get_vektor()

                # Get quadrant data
                quadran = Quadran(vektor_output)
                quadran_output = quadran.get_quadran()

                # Flatten to feature columns
                # quadran_output: [block_idx, x, y, theta, magnitude, quadran_label]
                for b_id, qd in enumerate(quadran_output, start=1):
                    row[f"{comp}_x{b_id}"] = int(qd[1])
                    row[f"{comp}_y{b_id}"] = int(qd[2])
                    row[f"{comp}_t{b_id}"] = float(qd[3])
                    row[f"{comp}_m{b_id}"] = float(qd[4])

            feature_rows.append(row)

        df = pd.DataFrame(feature_rows)
        return df

    def _extract_frames(self, video_input: bytes | str | Path) -> list[np.ndarray]:
        """
        Extract frames from video input.

        Args:
            video_input: Video bytes or path

        Returns:
            List of frames (BGR format)
        """
        temp_file = None

        try:
            if isinstance(video_input, bytes):
                # Save to temp file for OpenCV
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.write(video_input)
                temp_file.close()
                video_path = temp_file.name
            else:
                video_path = str(video_input)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()
            return frames

        finally:
            # Clean up temp file
            if temp_file is not None:
                import os

                os.unlink(temp_file.name)

    def predict(self, preprocessed_data: pd.DataFrame) -> dict[str, Any]:
        """
        Run TabR inference on preprocessed feature DataFrame.

        Args:
            preprocessed_data: DataFrame from preprocess() with POC-ABS features

        Returns:
            Dictionary with prediction results including:
            - prediction: Final predicted class (majority vote)
            - confidence: Average probability for predicted class
            - frame_predictions: Per-frame predictions
            - frame_probabilities: Per-frame probabilities
            - total_frames: Number of frames processed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.model.eval()

        # Use model's predict_from_df method
        pred, prob = self.model.predict_from_df(preprocessed_data, threshold=0.5)

        # Aggregate predictions (majority vote)
        unique, counts = np.unique(pred, return_counts=True)
        final_prediction = int(unique[np.argmax(counts)])

        # Calculate confidence as proportion of frames predicting the final class
        vote_confidence = float(np.max(counts) / len(pred))

        # Average probability for the positive class
        avg_prob = float(np.mean(prob))

        # Per-frame stats
        anxiety_ratio = float(np.mean(pred == 1))
        non_anxiety_ratio = float(np.mean(pred == 0))

        return {
            "prediction": final_prediction,
            "prediction_label": "anxiety" if final_prediction == 1 else "non_anxiety",
            "confidence": vote_confidence,
            "avg_probability": avg_prob,
            "total_frames": len(pred),
            "anxiety_frame_ratio": anxiety_ratio,
            "non_anxiety_frame_ratio": non_anxiety_ratio,
            "frame_predictions": pred.tolist(),
            "frame_probabilities": prob.tolist(),
        }

    def unload(self) -> None:
        """Unload model and artifacts from memory."""
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.context_x = None
        self.context_y = None
        self.roi_extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._is_loaded = False
