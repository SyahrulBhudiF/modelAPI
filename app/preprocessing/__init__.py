"""
Preprocessing Module

Contains components for video frame preprocessing including:
- POC (Phase-Only Correlation) for motion detection
- Vektor for motion vector extraction
- Quadran for motion direction classification
- ROIExtractor for facial region extraction using dlib
"""

from app.preprocessing.poc import POC
from app.preprocessing.quadran import Quadran
from app.preprocessing.roi_extractor import ROIExtractor
from app.preprocessing.vektor import Vektor

__all__ = ["POC", "Vektor", "Quadran", "ROIExtractor"]
