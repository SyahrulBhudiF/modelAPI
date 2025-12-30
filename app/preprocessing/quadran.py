"""
Quadran Module

Classifies motion vectors into quadrants based on direction.
Calculates angle (theta), magnitude, and quadrant labels for motion analysis.
"""

import numpy as np

from app.utils.helpers import format_number_and_round_numpy


class Quadran:
    """
    Motion quadrant classifier.

    Analyzes motion vectors and classifies them into quadrants
    based on their direction (angle from positive X-axis).

    Quadrants:
    - Q1: 0° - 90° (right-up)
    - Q2: 90° - 180° (left-up)
    - Q3: 180° - 270° (left-down)
    - Q4: 270° - 360° (right-down)
    """

    def __init__(self, coor_data: np.ndarray):
        """
        Initialize Quadran classifier.

        Args:
            coor_data: Output from Vektor.get_vektor() with shape (num_blocks, 6)
                      Column 4: oX (relative X direction)
                      Column 5: oY (relative Y direction)
        """
        self.data_a = coor_data[:, 4]  # oX - X direction
        self.data_b = coor_data[:, 5]  # oY - Y direction

    def get_quadran(self) -> np.ndarray:
        """
        Classify motion vectors into quadrants.

        Returns:
            Array of shape (num_blocks, 6) with dtype=object containing:
                - Column 0: Block index (string)
                - Column 1: X direction (int)
                - Column 2: Y direction (int)
                - Column 3: Theta angle in degrees (float, rounded)
                - Column 4: Magnitude (float, rounded)
                - Column 5: Quadrant label (string)
        """
        quadran_data = np.empty((len(self.data_a), 6), dtype=object)

        for i in range(len(self.data_a)):
            x = int(self.data_a[i])
            y = int(self.data_b[i])

            # Calculate angle (theta) in degrees
            # Add 360 if Y < 0 to keep angle in range [0, 360)
            theta = np.degrees(np.arctan2(y, x)) + 360 * (y < 0)

            # Calculate magnitude using Pythagorean theorem
            magnitude = np.sqrt(x**2 + y**2)

            # Classify into quadrant
            if x == 0 and y == 0:
                quadran_label = "No Quadran X Y = 0"
            elif 0 <= theta < 90:
                quadran_label = "Q1"  # Right-up
            elif 90 <= theta < 180:
                quadran_label = "Q2"  # Left-up
            elif 180 <= theta < 270:
                quadran_label = "Q3"  # Left-down
            elif 270 <= theta < 360:
                quadran_label = "Q4"  # Right-down
            else:
                quadran_label = "No Quadran"

            quadran_data[i, :] = [
                str(i),  # Block index
                x,  # X direction
                y,  # Y direction
                format_number_and_round_numpy(theta),  # Theta (degrees)
                format_number_and_round_numpy(magnitude),  # Magnitude
                quadran_label,  # Quadrant label
            ]

        return quadran_data

    # Alias for backward compatibility
    getQuadran = get_quadran
