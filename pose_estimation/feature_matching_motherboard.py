import cv2

from config.config import PoseEstimationConfig
from utils.utils import find_latest_file_in_latest_directory


class FeatureMatchingMotherboard:
    def __init__(self):
        self.pose_est_cfg = PoseEstimationConfig().parse()
        self.ref_points = []
        self.image_points = []
        self.reduced_ref_points = []
        self.points_to_append = []
        self.cropping = False
        self.detecting_image = find_latest_file_in_latest_directory()
        self.template_image = None
        self.resized = None
        self.number_of_points = 7
        self.found = []

    def click_and_select(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points = [x * self.pose_est_cfg.size_coefficient, y * self.pose_est_cfg.size_coefficient]
            self.image_points = [(x * self.pose_est_cfg.size_coefficient, y * self.pose_est_cfg.size_coefficient)]
            self.reduced_ref_points = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_points.append((x * self.pose_est_cfg.size_coefficient, y * self.pose_est_cfg.size_coefficient))
            self.image_points.append((x * self.pose_est_cfg.size_coefficient, y * self.pose_est_cfg.size_coefficient))
            self.reduced_ref_points.append((x, y))
            self.cropping = False

            cv2.rectangle(self.resized, self.reduced_ref_points[0], self.reduced_ref_points[1], (0, 255, 0), 2)
            cv2.imshow("image", self.resized)

    def capture(self):
        self.resized = cv2.resize(self.detecting_image,
                                  dsize=(
                                      self.detecting_image[1] // self.pose_est_cfg.size_coefficient,
                                      self.detecting_image[0] // self.pose_est_cfg.size_coefficient
                                  ))

        clone = self.detecting_image.copy()

        for i in range(self.number_of_points):
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("image", self.click_and_select)

            while True:
                cv2.imshow("image", self.resized)
                key = cv2.waitKey(1)

                if key == ord("c"):
                    break

            if len(self.ref_points) == 2:
                roi = clone[(self.ref_points[0][1]):
                            (self.ref_points[1][1]),
                            (self.ref_points[0][0]):
                            (self.ref_points[1][0])]

                cv2.imshow("roi", roi)
                filename = f"{i}"

                cv2.imwrite(filename, roi)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def main(self):
        self.capture()


if __name__ == "__main__":
    feature_matching = FeatureMatchingMotherboard()
    feature_matching.main()
