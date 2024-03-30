import cv2
import json
import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, List, Any

from config.config_selector import camera_config, feature_matching_config
from config.config import FeatureMatchingConfig
from utils.utils import (create_dir, create_timestamp, find_latest_file_in_latest_directory, find_latest_subdir,
                         file_reader, measure_execution_time, NumpyEncoder)


class FeatureMatchingMotherboard:
    def __init__(self, create_roi_images):
        timestamp = create_timestamp()
        self.feature_matching_cfg = FeatureMatchingConfig().parse()

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
                create_dir(root_dir=feature_matching_config().get("roi_images"),
                           timestamp=timestamp)
            )
        else:
            self.roi_images_path = find_latest_subdir(feature_matching_config().get("roi_images"))

        self.pairing_images_path = (
            create_dir(feature_matching_config().get("pairing_images"),
                       timestamp)
        )
        self.mounting_hole_center_points_path = (
            create_dir(feature_matching_config().get("mounting_hole_center_points"),
                       timestamp)
        )

    def click_and_select(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Handle mouse events for selecting points on the image.

        Args:
            event: The event type (e.g., left button down, left button up).
            x: The x-coordinate of the mouse cursor.
            y: The y-coordinate of the mouse cursor.
            flags: Any flags passed by the OpenCV event handler.
            param: Any additional parameters passed by the OpenCV event handler.

        Return:
            None
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points = [(x * self.feature_matching_cfg.size_coefficient,
                                y * self.feature_matching_cfg.size_coefficient)]
            self.image_points = [(x * self.feature_matching_cfg.size_coefficient,
                                  y * self.feature_matching_cfg.size_coefficient)]
            self.reduced_ref_points = [(x, y)]
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_points.append((x * self.feature_matching_cfg.size_coefficient,
                                    y * self.feature_matching_cfg.size_coefficient))
            self.image_points.append((x * self.feature_matching_cfg.size_coefficient,
                                      y * self.feature_matching_cfg.size_coefficient))
            self.reduced_ref_points.append((x, y))
            self.cropping = False

            cv2.rectangle(self.resized, self.reduced_ref_points[0], self.reduced_ref_points[1], (0, 255, 0), 2)
            cv2.imshow("image", self.resized)

    def capture(self):
        """
        Capture regions of interest (ROIs) from the motherboard image.

        Returns:
            None
        """

        self.resized = cv2.resize(src=self.motherboard_image,
                                  dsize=(
                                      self.motherboard_image.shape[1] // self.feature_matching_cfg.size_coefficient,
                                      self.motherboard_image.shape[0] // self.feature_matching_cfg.size_coefficient
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

    def sift_detector(self, sift: Any, grayscale_roi_img: Any, grayscale_motherboard_img: Any) \
            -> Tuple[List[Any], Any, Any]:
        """
        Detect keypoints and compute descriptors for the given grayscale images using SIFT.

        Args:
            sift: The SIFT object used for keypoint detection and descriptor computation.
            grayscale_roi_img: The grayscale image of the region of interest.
            grayscale_motherboard_img: The grayscale image of the motherboard.

        Returns:
            A tuple containing:
                - A list of good matches between the descriptors.
                - Key points detected in the region of interest image.
                - Key points detected in the motherboard image.
        """

        key_points1, descriptors1 = sift.detectAndCompute(grayscale_roi_img, None)
        key_points2, descriptors2 = sift.detectAndCompute(grayscale_motherboard_img, None)

        bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        matches = bf.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)

        # Lowe's test
        good_matches = []
        for i, j in matches:
            if i.distance < self.feature_matching_cfg.threshold * j.distance:
                good_matches.append(i)

        return good_matches, key_points1, key_points2

    def find_mounting_holes(self, roi_img: Any, good_matches: List[Any], key_points1: Any, key_points2: Any,
                            idx: int) -> List[int]:
        """
        Find mounting holes in the region of interest image.

        Args:
            roi_img: The region of interest image.
            good_matches: List of good matches between key-points.
            key_points1: Key-points detected in the region of interest image.
            key_points2: Key-points detected in the motherboard image.
            idx: Index of the ROI image.

        Returns:
            List of integers indicating matches mask.
        """

        source_points = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        ret, mask = cv2.findHomography(srcPoints=source_points,
                                       dstPoints=destination_points,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=5.0)

        matches_mask = mask.ravel().tolist()

        blured_roi_img = cv2.GaussianBlur(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY),
                                          self.feature_matching_cfg.kernel_size,
                                          0)

        circles = cv2.HoughCircles(blured_roi_img,
                                   cv2.HOUGH_GRADIENT,
                                   dp=self.feature_matching_cfg.dp,
                                   minDist=self.feature_matching_cfg.minDist,
                                   param1=self.feature_matching_cfg.param1,
                                   param2=self.feature_matching_cfg.param2,
                                   minRadius=self.feature_matching_cfg.minRadius,
                                   maxRadius=self.feature_matching_cfg.maxRadius)

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

    def save_pairing_images(self, idx: int, roi_img: Any, key_points1: Any, key_points2: Any, matches_mask: List[int],
                            good_matches: List[Any]) -> None:
        """
        Save the paired images with matches drawn between key-points.

        Args:
            idx: Index of the paired images.
            roi_img: The region of interest image.
            key_points1: Key-points detected in the region of interest image.
            key_points2: Key-points detected in the motherboard image.
            matches_mask: List indicating matches mask.
            good_matches: List of good matches between key-points.

        Returns:
            None
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

    def process_roi(self, idx: int, img: str, sift, results: list) -> None:
        """
        Process a region of interest (ROI) image.

        Args:
            idx: Index of the ROI image.
            img: Path to the ROI image file.
            sift: SIFT object
            results: list to store results

        Returns:
            None
        """

        roi_img = cv2.imread(img, cv2.IMREAD_COLOR)
        grayscale_roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        grayscale_motherboard_img = cv2.cvtColor(self.motherboard_image, cv2.COLOR_BGR2GRAY)

        good_matches, key_points1, key_points2 = self.sift_detector(sift,
                                                                    grayscale_roi_img,
                                                                    grayscale_motherboard_img)

        if len(good_matches) > self.feature_matching_cfg.min_match_count:
            matches_mask = self.find_mounting_holes(roi_img, good_matches, key_points1, key_points2, idx)
        else:
            error_message = (
                f"Not enough matches found: {len(good_matches)}, "
                f"minimum good matches threshold is: {self.feature_matching_cfg.min_match_count}"
            )
            logging.error(error_message)
            raise ValueError(error_message)

        self.save_pairing_images(idx, roi_img, key_points1, key_points2, matches_mask, good_matches)

        # Append result in the order of processing
        results.append((idx, good_matches, key_points1, key_points2, matches_mask))

    @measure_execution_time
    def matching(self) -> None:
        """
        Process ROI images using ThreadPoolExecutor for parallel execution.
        """

        roi_images = file_reader(self.roi_images_path, extension="jpg")
        sift = cv2.SIFT_create(nfeatures=self.feature_matching_cfg.nfeatures,
                               nOctaveLayers=self.feature_matching_cfg.nOctaveLayers,
                               contrastThreshold=self.feature_matching_cfg.contrastThreshold,
                               edgeThreshold=self.feature_matching_cfg.edgeThreshold,
                               sigma=self.feature_matching_cfg.sigma)

        # List to store results in the order of processing
        results = []

        # Process ROI images in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(self.process_roi, idx, img, sift, results) for idx, img in enumerate(roi_images)]

            # Wait for all futures to complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing ROI images"):
                future.result()

        results.sort(key=lambda x: x[0])

        if results:
            for idx, good_matches, key_points1, key_points2, matches_mask in results:
                roi_img = cv2.imread(roi_images[idx], cv2.IMREAD_COLOR)
                self.find_mounting_holes(roi_img, good_matches, key_points1, key_points2, idx)

        if len(self.points_to_append) > 2:
            self.save_points()

    def save_points(self) -> None:
        """
        Save the detected mounting hole center points to a JSON file.

        Return:
            None
        """

        output_json_filename = os.path.join(self.mounting_hole_center_points_path, "point_pairs.json")

        processed_points = {}
        for key, value in self.points_to_append.items():
            processed_points[key] = [coord[0] for coord in value]

        processed_points = dict(sorted(processed_points.items()))
        processed_points = [[int(a) for a in b[0]] for b in processed_points.values()]

        with open(output_json_filename, "w") as json_file:
            json.dump(processed_points, json_file, cls=NumpyEncoder)

    def main(self) -> None:
        """

        Returns:
             None
        """

        if self.create_roi_images:
            self.capture()
        self.matching()


if __name__ == "__main__":
    try:
        feature_matching = FeatureMatchingMotherboard(create_roi_images=True)
        feature_matching.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
