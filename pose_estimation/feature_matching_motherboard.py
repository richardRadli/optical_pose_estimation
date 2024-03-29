import cv2
import json
import logging
import numpy as np
import os

from tqdm import tqdm

from config.config_selector import camera_config, pose_estimation_config
from config.config import PoseEstimationConfig
from utils.utils import (create_dir, create_timestamp, find_latest_file_in_latest_directory, find_latest_subdir,
                         file_reader, measure_execution_time, NumpyEncoder)


class FeatureMatchingMotherboard:
    def __init__(self, create_roi_images):
        timestamp = create_timestamp()
        self.pose_est_cfg = PoseEstimationConfig().parse()

        self.create_roi_images = create_roi_images

        self.ref_points = []
        self.image_points = []
        self.reduced_ref_points = []
        self.points_to_append = {}

        self.cropping = False
        self.template_image = None
        self.resized = None

        self.motherboard_image = cv2.imread((
            find_latest_file_in_latest_directory(camera_config().get("undistorted_motherboard_images"))
        ))

        self.number_of_points = 7

        if create_roi_images:
            self.roi_images_path = (
                create_dir(root_dir=pose_estimation_config().get("roi_images"),
                           timestamp=timestamp)
            )
        else:
            self.roi_images_path = find_latest_subdir(pose_estimation_config().get("roi_images"))

        self.pairing_images_path = (
            create_dir(pose_estimation_config().get("pairing_images"),
                       timestamp)
        )
        self.mounting_hole_center_points_path = (
            create_dir(pose_estimation_config().get("mounting_hole_center_points"),
                       timestamp)
        )

    def click_and_select(self, event, x, y, flags, param):
        """
        
        Args
            event:
            x:
            y:
            flags:
            param:

        Return:
            None
        """
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points = [(x * self.pose_est_cfg.size_coefficient, y * self.pose_est_cfg.size_coefficient)]
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
        """

        Returns:

        """
        
        self.resized = cv2.resize(src=self.motherboard_image,
                                  dsize=(
                                      self.motherboard_image.shape[1] // self.pose_est_cfg.size_coefficient,
                                      self.motherboard_image.shape[0] // self.pose_est_cfg.size_coefficient
                                  ))
        clone = self.motherboard_image.copy()

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

                output_filename = os.path.join(self.roi_images_path, f"roi_{i}.jpg")
                cv2.imwrite(output_filename, roi)
                logging.info(f"Roi image {i}. saved to {output_filename}")

            cv2.destroyAllWindows()

    @staticmethod
    @measure_execution_time
    def sift_detector(sift, grayscale_roi_img, grayscale_motherboard_img, threshold: float = 0.6):
        """

        Args:
            sift:
            grayscale_roi_img:
            grayscale_motherboard_img:
            threshold:
        Returns:

        """

        key_points1, descriptors1 = sift.detectAndCompute(grayscale_roi_img, None)
        key_points2, descriptors2 = sift.detectAndCompute(grayscale_motherboard_img, None)

        bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        matches = bf.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)

        # Lowe's test
        good_matches = []
        for i, j in matches:
            if i.distance < threshold * j.distance:
                good_matches.append(i)

        return good_matches, key_points1, key_points2

    @measure_execution_time
    def find_mounting_holes(self, roi_img, good_matches, key_points1, key_points2, idx):
        """
        
        Args:
            roi_img:
            good_matches:
            key_points1:
            key_points2:
            idx:

        Returns:

        """
        source_points = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        ret, mask = cv2.findHomography(srcPoints=source_points,
                                       dstPoints=destination_points,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=5.0)

        matches_mask = mask.ravel().tolist()
        blured_roi_img = cv2.GaussianBlur(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        circles = cv2.HoughCircles(blured_roi_img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=15, maxRadius=40)

        center = 0, 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = i[0], i[1]
                cv2.circle(roi_img, center, 1, (255, 0, 100), 3)
                radius = i[2]
                cv2.circle(roi_img, center, radius, (255, 0, 100), 3)

            point_to_transform = np.float32([center]).reshape(-1, 1, 2)
            transformed_point = cv2.perspectiveTransform(point_to_transform, ret)
            self.points_to_append[idx] = transformed_point
        else:
            raise ValueError("No circles detected")

        return matches_mask

    def save_pairing_images(self, idx, roi_img, key_points1, key_points2, matches_mask, good_matches):
        """

        Args:
            idx:
            roi_img:
            key_points1:
            key_points2:
            matches_mask:
            good_matches:
        
        Returns:
         
        """
        
        draw_parameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        result = cv2.drawMatches(img1=roi_img,
                                 keypoints1=key_points1,
                                 img2=self.motherboard_image,
                                 keypoints2=key_points2,
                                 matches1to2=good_matches,
                                 outImg=None,
                                 **draw_parameters)
        pairing_filename = os.path.join(self.pairing_images_path, f"pairing_{idx}.jpg")
        cv2.imwrite(pairing_filename, result)

    def matching(self, min_match_count: int = 25):
        roi_images = file_reader(self.roi_images_path, extension="jpg")
        sift = cv2.SIFT_create(nfeatures=0,
                               nOctaveLayers=5,
                               contrastThreshold=0.05,
                               edgeThreshold=40,
                               sigma=1.0)

        for idx, img in tqdm(enumerate(roi_images), total=len(roi_images), desc="Matching images"):
            roi_img = cv2.imread(img, cv2.IMREAD_COLOR)
            grayscale_roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            grayscale_motherboard_img = cv2.cvtColor(self.motherboard_image, cv2.COLOR_BGR2GRAY)

            good_matches, key_points1, key_points2 = self.sift_detector(sift,
                                                                        grayscale_roi_img,
                                                                        grayscale_motherboard_img)

            if len(good_matches) > min_match_count:
                matches_mask = self.find_mounting_holes(roi_img, good_matches, key_points1, key_points2, idx)
            else:
                error_message = \
                    (f"Not enough matches found: {len(good_matches)}, "
                     f"minimum good matches threshold is: {min_match_count}")
                logging.error(error_message)
                raise ValueError(error_message)

            self.save_pairing_images(idx, roi_img, key_points1, key_points2, matches_mask, good_matches)

        if len(self.points_to_append) > 2:
            self.save_points()

    def save_points(self):
        """

        Return:
        """

        output_json_filename = os.path.join(self.mounting_hole_center_points_path, "point_pairs.json")

        processed_points = {}
        for key, value in self.points_to_append.items():
            processed_points[key] = [coord[0] for coord in value]

        # Save the processed data to a JSON file
        with open(output_json_filename, "w") as json_file:
            json.dump(processed_points, json_file, cls=NumpyEncoder)

    def main(self):
        if self.create_roi_images:
            self.capture()
        self.matching()


if __name__ == "__main__":
    try:
        feature_matching = FeatureMatchingMotherboard(create_roi_images=False)
        feature_matching.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
