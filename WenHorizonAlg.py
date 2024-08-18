import cv2 as cv
import numpy as np
import os
from time import time
from warnings import warn
from math import pi, atan, sin, cos


class WenL:
    def __init__(self, ksize=5):
        self.ksize = ksize
        self.hough_D_rho = 2
        self.hough_D_theta = pi / 180
        self.D_theta = 1 * (pi / 180)
        self.D_rho = 1

        self.phi = np.nan
        self.Y = np.nan

        self.color_red = (0, 0, 255)
        self.color_blue = (255, 0, 0)
        self.color_green = (0, 255, 0)
        self.color_yellow = (0, 255, 255)
        self.color_aqua = (255, 255, 0)
        self.all_colors = [self.color_red, self.color_blue, self.color_green, self.color_yellow, self.color_aqua]

    def get_horizon(self, img):
        self.start_time = time()
        self.F_det = True  # will be set to false if no edge point is detected, which implies that no horizon is
        # detected
        self.get_horizon_edges(img=img)
        self.hough_transform()
        self.linear_least_square_fitting()
        # self.outlier_hl_handler()  # check if the detection is outlier

        if self.F_det:  # we check this flag again because it can be changed in self.outlier_hl_handler()
            print("Y = {}, phi = {}".format(self.Y, self.phi))
            self.end_time = time()
            self.latency = round((self.end_time - self.start_time), 4)
        else:
            self.img_edges = np.zeros(shape=self.in_img_gray.shape,
                                      dtype=np.uint8)  # create an edge map with no edge point
            self.Y = np.nan
            self.phi = np.nan
            self.latency = np.nan
            self.img_with_hl = img
        print("time execution is: ", self.latency, " seconds")
        return self.Y, self.phi, self.latency, self.F_det

    def get_horizon_edges(self, img):

        self.org_height = img.shape[0]
        self.org_width = img.shape[1]

        self.in_img_bgr = img
        self.img_with_hl = self.in_img_bgr.copy()
        self.in_img_gray = cv.cvtColor(self.in_img_bgr, cv.COLOR_RGB2GRAY)

        # Calculate the Sobel gradient response
        self.sobelx = cv.Sobel(self.in_img_gray, cv.CV_64F, 1, 0, ksize=self.ksize)
        self.sobely = cv.Sobel(self.in_img_gray, cv.CV_64F, 0, 1, ksize=self.ksize)
        self.soblel_mag, _ = cv.cartToPolar(self.sobelx, self.sobely)  # _ don't care about the angle
        # normalizing self.soblel_mag into [0, 255] range
        self.soblel_mag = np.uint8((np.multiply(self.soblel_mag, np.divide(255, np.amax(self.soblel_mag)))))
        # get the OTSU's threshold
        self.otsu_th, _ = cv.threshold(self.soblel_mag, 0, 255,
                                       cv.THRESH_BINARY + cv.THRESH_OTSU)  # returns Otsu_threshold, thresholded_image
        self.soblel_mag = np.int16(self.soblel_mag)
        self.img_edges = cv.Canny(self.soblel_mag, self.soblel_mag, self.otsu_th, self.otsu_th)

    def hough_transform(self):
        if not self.F_det:  # True if the edge map contains edges. False otherwise (zero edges detected)
            return
        self.hough_lines = cv.HoughLines(image=self.img_edges, rho=self.hough_D_rho, theta=self.hough_D_theta,
                                         threshold=2, min_theta=np.pi / 3, max_theta=np.pi * (2 / 3))
        if self.hough_lines is not None:  # executes if Hough detects a line
            self.F_det = True
        else:  # True if no line is detected from the Hough transform
            self.phi = np.nan
            self.Y = np.nan
            self.latency = np.nan
            self.F_det = False

    def linear_least_square_fitting(self):
        if self.F_det:
            self.get_inlier_edges()
            self.inlier_edges_xy = np.zeros((self.inlier_edges_x.size, 2), dtype=np.int32)
            self.inlier_edges_xy[:, 0], self.inlier_edges_xy[:, 1] = self.inlier_edges_x, self.inlier_edges_y
            # print("shape of inlier edges: ", self.inlier_edges_xy.shape)
            [vx, vy, x, y] = cv.fitLine(points=self.inlier_edges_xy, distType=cv.DIST_L2,
                                        param=0, reps=1, aeps=0.01)
            self.hl_slope = float(vy / vx)  # float to convert from (1,) float numpy array to python float
            self.hl_intercept = float(y - self.hl_slope * x)

            self.xs_hl = int(0)
            self.xe_hl = int(self.org_width - 1)
            self.ys_hl = int(self.hl_intercept)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
            self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)

            self.phi = (-atan(self.hl_slope)) * (180 / pi)  # - because the y axis of images goes down
            self.Y = ((((self.org_width - 1) / 2) * self.hl_slope + self.hl_intercept))

    def get_inlier_edges(self):
        """
        Process is described in inlier_edges.pdf file attached with this code project.
        """
        self.rho, self.theta = self.hough_lines[0][0]  # self.theta in radians
        self.y_j, self.x_j = np.where(self.img_edges == 255)
        theta_p = self.theta + self.D_theta
        theta_n = self.theta - self.D_theta
        self.x_cte = 0.5 * (np.cos(theta_p) - np.cos(theta_n))
        self.y_cte = 0.5 * (np.sin(theta_p) - np.sin(theta_n))

        self.D_rho_j = np.abs(np.add(np.multiply(self.x_j, self.x_cte), np.multiply(self.y_j, self.y_cte)))
        self.D_rho_g = np.add(self.D_rho_j, self.D_rho)

        self.rho_j = np.add(np.multiply(self.x_j, np.cos(self.theta)), np.multiply(self.y_j, np.sin(self.theta)))
        inlier_condition = np.logical_and(self.rho_j <= (self.rho + self.D_rho_g / 2),
                                          self.rho_j >= (self.rho - self.D_rho_g / 2))

        self.inlier_edges_indexes = np.where(inlier_condition)
        self.inlier_edges_x = self.x_j[self.inlier_edges_indexes]
        self.inlier_edges_y = self.y_j[self.inlier_edges_indexes]
        self.inlier_edges_map = np.zeros(shape=self.img_edges.shape, dtype=np.uint8)
        self.inlier_edges_map[self.inlier_edges_y, self.inlier_edges_x] = 255

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds
        between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For instance, if
        the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform is sqrt(640^2 +
        480^2) * hough_threshold_ratio, rounded to the nearest integer.
        :param src_gt_folder: absolute path to the ground truth horizons corresponding to source video files.
        :param src_video_folder: absolute path to folder containing source video files to process
        :param dst_video_folder: absolute path where video files with drawn horizon will be saved.
        :param dst_quantitative_results_folder: destination folder where quantitative results will be saved.
        :param draw_and_save: if True, all detected horizons will be drawn on their corresponding frames and saved as video files
        in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            # yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
            #            " to proceed, otherwise, click on 'n'.\n"
            #            "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
            #            "files with similar names.")
            yn = 'y'
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "Wen.Li_" + src_video_name)
                if draw_and_save:
                    video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                                  True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                warning_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print("----------WARNING---------")
                print(warning_text_1)
                print("--------------------------")

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                # print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                #       format(self.Y, self.gt_position_hl, self.phi, self.gt_tilt_hl))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.Y,
                                                   self.phi,
                                                   round(abs(self.Y - self.gt_position_hl), 4),
                                                   round(abs(self.phi - self.gt_tilt_hl), 4),
                                                   self.latency]
                if draw_and_save:
                    self.draw_hl()  # draws the horizon on self.img_with_hl
                    video_writer.write(self.img_with_hl)
            cap.release()
            if draw_and_save:
                video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder, src_video_name_no_ext+".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder, "all_det_hl_"+src_video_folder_name+".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

        self.Y_hl_all = self.det_horizons_all_files[:, 2]
        self.alpha_hl_all = self.det_horizons_all_files[:, 3]
        self.latency_all = self.det_horizons_all_files[:, 4]
        self.false_positive_nbr = np.size(np.argwhere(np.isnan(self.Y_hl_all)))

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl'
        """
        if self.F_det:
            thickness = 5
            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255),
                    thickness=thickness)

