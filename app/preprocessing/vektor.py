"""
Vektor Module

Extracts motion vectors from POC (Phase-Only Correlation) output.
Calculates displacement vectors between consecutive frames.
"""

import numpy as np


class Vektor:
    """
    Motion vector extractor from POC output.

    Analyzes POC results to determine motion direction and magnitude
    for each block in the frame.
    """

    def __init__(self, poc_output: list, block_size: int):
        """
        Initialize Vektor extractor.

        Args:
            poc_output: Output from POC.getPOC() containing [poc_values, coor_awal, rect]
            block_size: Size of each block (e.g., 7 for 7x7 blocks)
        """
        self.poc = poc_output[0]  # POC values for each block
        self.coor_awal = poc_output[1]  # Starting coordinates for each block
        self.block_size = block_size

    def get_vektor(self) -> np.ndarray:
        """
        Extract motion vectors from POC output.

        Returns:
            Array of shape (num_blocks, 6) containing:
                - Column 0-1: Starting point (x, y)
                - Column 2-3: Vector components (delta_x, delta_y)
                - Column 4-5: Relative direction (oX, oY)
        """
        mb_x = self.block_size
        mb_y = self.block_size

        cur_x = np.arange(0, mb_x)
        cur_y = np.arange(0, mb_y)

        # Calculate center reference point
        nil_teng = np.int16(np.median(cur_x))
        med_x = nil_teng + 1
        med_y = nil_teng + 1

        # Direction representation arrays
        rep_x = np.arange(-(nil_teng), med_x)
        rep_y = np.arange(nil_teng, -(med_y), -1)

        # Output array: [x_start, y_start, delta_x, delta_y, oX, oY]
        output = np.zeros((len(self.coor_awal), 6))

        val_poc = self.poc

        for i in range(val_poc.shape[2]):
            r = val_poc[:, :, i]

            # Find position of maximum correlation (most similar motion)
            temp_y, temp_x = np.where(r == np.max(r))

            # Handle ambiguity: if multiple peaks found, use center
            if len(temp_y) > 1 or len(temp_x) > 1:
                temp_x = nil_teng
                temp_y = nil_teng
            else:
                temp_x = temp_x[0]
                temp_y = temp_y[0]

                # Check if there's actual motion (not at center)
                if temp_x != nil_teng or temp_y != nil_teng:
                    cor_x = self.coor_awal[i][0]  # Block X coordinate
                    cor_y = self.coor_awal[i][1]  # Block Y coordinate

                    t_x = cor_x - med_x
                    t_y = cor_y - med_y

                    # Relative motion direction
                    o_x = rep_x[cur_x[temp_x]]
                    o_y = rep_y[cur_y[temp_y]]

                    # Destination coordinates
                    m_x = cor_x - (mb_x - temp_x)
                    m_y = cor_y - (mb_y - temp_y)

                    # Calculate vector as difference between start and end points
                    p1 = [t_x, t_y]
                    p2 = [m_x, m_y]
                    v = np.array(p2) - np.array(p1)

                    output[i, 0] = p1[0]  # x start
                    output[i, 1] = p1[1]  # y start
                    output[i, 2] = v[0]  # delta x
                    output[i, 3] = v[1]  # delta y
                    output[i, 4] = o_x  # relative x direction
                    output[i, 5] = o_y  # relative y direction

        return output

    # Alias for backward compatibility
    getVektor = get_vektor
