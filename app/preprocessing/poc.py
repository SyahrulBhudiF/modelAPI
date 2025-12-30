"""
Phase-Only Correlation (POC) Module

Implements Phase-Only Correlation for motion estimation between image blocks.
Used for detecting micro-movements in video frames.
"""

import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2


class POC:
    """
    Phase-Only Correlation calculator.

    Computes the phase correlation between blocks of two consecutive frames
    to detect motion patterns.
    """

    def __init__(
        self, img_block_cur: np.ndarray, img_block_ref: np.ndarray, block_size: int
    ):
        """
        Initialize POC calculator.

        Args:
            img_block_cur: Current frame (grayscale numpy array)
            img_block_ref: Reference frame (grayscale numpy array)
            block_size: Size of the blocks for POC computation
        """
        self.img_block_cur = img_block_cur
        self.img_block_ref = img_block_ref
        self.block_size = block_size

    def hann_calc(self) -> np.ndarray:
        """
        Calculate 2D Hanning window.

        Returns:
            2D Hanning window of size (block_size, block_size)
        """
        window = np.hanning(self.block_size)
        window = np.outer(window, window)
        return window

    def calc_poc(
        self,
        block_ref: np.ndarray,
        block_curr: np.ndarray,
        window: np.ndarray,
        mb_x: int,
        mb_y: int,
    ) -> np.ndarray:
        """
        Calculate Phase-Only Correlation between two blocks.

        Args:
            block_ref: Reference block
            block_curr: Current block
            window: Hanning window for windowing
            mb_x: Block width
            mb_y: Block height

        Returns:
            POC result matrix showing correlation peaks
        """
        # Apply Hanning window and compute FFT
        fft_ref = fft2(block_ref * window, (mb_x, mb_y))
        fft_curr = fft2(block_curr * window, (mb_x, mb_y))

        # Compute phase correlation
        R1 = fft_ref * np.conj(fft_curr)

        # Compute magnitude (avoid division by zero)
        R2 = np.abs(R1)
        R2[R2 == 0] = 1e-31

        # Normalize by magnitude (phase-only)
        R = R1 / R2

        # Inverse FFT to get correlation
        r = ifft2(R)
        r = np.abs(r)

        # Shift zero-frequency to center
        r = fftshift(r)

        return r

    def get_poc(self) -> list:
        """
        Compute POC for all blocks in the image pair.

        Returns:
            List containing:
            - poc: 3D array of POC values for each block (mb_y, mb_x, num_blocks)
            - coor_awal: Starting coordinates for each block
            - rect: Rectangle coordinates (x, y, width, height) for each block
        """
        mb_x = self.block_size
        mb_y = self.block_size

        # Calculate Hanning window
        window = self.hann_calc()

        img0 = self.img_block_cur
        img1 = self.img_block_ref

        # Convert to int
        cols, rows = img0.shape
        img0 = img0.astype(int)
        img1 = img1.astype(int)

        # Calculate number of blocks
        cols_y = int(np.floor(cols / mb_y))
        rows_x = int(np.floor(rows / mb_x))

        # Initialize block storage
        blocks_curr = np.empty((cols_y, rows_x), dtype=object)
        blocks_ref = np.empty((cols_y, rows_x), dtype=object)

        # Calculate remainder pixels
        mod_y = cols % mb_y
        mod_x = rows % mb_x

        # Initialize output arrays
        poc = np.zeros((mb_y, mb_x, cols_y * rows_x))
        coor_awal = np.zeros((cols_y * rows_x, 2))
        rect = np.zeros((cols_y * rows_x, 4))

        nm = 0
        n_y = 0
        n_yy = 1

        # Iterate through blocks
        for y in range(0, cols - mod_y, mb_y):
            n_x = 0
            n_xx = 1
            for x in range(0, rows - mod_x, mb_x):
                # Extract blocks
                blocks_curr[n_y, n_x] = img0[y : y + mb_y, x : x + mb_x]
                blocks_ref[n_y, n_x] = img1[y : y + mb_y, x : x + mb_x]

                # Store rectangle coordinates
                rect[nm, :] = [x, y, mb_x, mb_y]

                block_ref = blocks_ref[n_y, n_x]
                block_curr = blocks_curr[n_y, n_x]

                # Calculate POC for this block pair
                r = self.calc_poc(block_ref, block_curr, window, mb_x, mb_y)
                poc[:, :, nm] = r

                # Store starting coordinates
                coor_awal[nm, 0] = n_xx * mb_x
                coor_awal[nm, 1] = n_yy * mb_y

                n_x += 1
                n_xx += 1
                nm += 1

            n_y += 1
            n_yy += 1

        return [poc, coor_awal, rect]
