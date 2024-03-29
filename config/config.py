import argparse
import os


class CameraAndCalibrationConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--size_coeff', type=int, default=3,
                                 help="The shown image will be resized by the given coefficient.")
        self.parser.add_argument('--height', type=int, default=2448, help="Height of the image.")
        self.parser.add_argument('--width', type=int, default=3264, help="Width of the image.")
        self.parser.add_argument('--camID', type=int, default=0, help="Default camera device index")
        self.parser.add_argument('--chs_col', type=int, default=15, help="Number of columns in the chessboard")
        self.parser.add_argument('--chs_row', type=int, default=10, help="Number of rows in the chessboard")
        self.parser.add_argument('--square_size', type=int, default=25, help="Square size of the chessboard")
        self.parser.add_argument('--error_threshold', type=float, default=0.2,
                                 help="Error threshold for the calibration")
        self.parser.add_argument('--size_coefficients', type=int, default=3)
        self.parser.add_argument('--undistort', type=bool, default=False)
        self.parser.add_argument('--cpu_threads', type=int, default=os.cpu_count(),
                                 help="Number of CPU threads")
        self.parser.add_argument('--operation', type=str, default='calibration_images',
                                 choices=["calibration_images", "chessboard_images", "motherboard_images"])

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class FeatureMatchingConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--size_coefficient', type=int, default=1)

        self.parser.add_argument('--kernel_size', type=tuple, default=(5, 5))
        self.parser.add_argument('--dp', type=int, default=1)
        self.parser.add_argument("--minDist", type=int, default=20)
        self.parser.add_argument('--param1', type=int, default=50)
        self.parser.add_argument("--param2", type=int, default=30)
        self.parser.add_argument("--minRadius", type=int, default=15)
        self.parser.add_argument("--maxRadius", type=int, default=40)

        self.parser.add_argument("--nfeatures", type=int, default=0)
        self.parser.add_argument("--nOctaveLayers", type=int, default=5)
        self.parser.add_argument("--contrastThreshold", type=float, default=0.05)
        self.parser.add_argument("--edgeThreshold", type=int, default=40)
        self.parser.add_argument("--sigma", type=float, default=1.0)
        self.parser.add_argument("--min_match_count", type=int, default=25)
        self.parser.add_argument("--threshold", type=float, default=0.6)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
