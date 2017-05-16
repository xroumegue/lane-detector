import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os
import numpy as np;
import cv2;
from itertools import accumulate
from random import choices
from ransac import ransacModel, getModel, doRansac
from laneDetector.lines import line
from sys import version_info
from sympy import tan, sin, cos, atan2, sqrt, Point, Line, pi, Polygon

RANSAC_LINE_DEFAULT_BOUNDING_BOX_WIDTH = 20
RANSAC_LINE_DEFAULT_SAMPLES = 5
RANSAC_LINE_DEFAULT_ITERATIONS = 50
RANSAC_LINE_DEFAULT_THRESHOLD = 0.3
RANSAC_LINE_DEFAULT_GOOD = 10
if version_info.major < 3 and version_info.minor < 6:
    raise ValueError("Need to run with python version >= 3.6.x")

class npSvdModel:
    def __init__(self, logger, debug=False):
        self.debug = debug
        self.logger= logger


    def random_partition(self, n, data):
        all_idxs = np.arange(data.shape[0])
        w = list(accumulate(data[:, -1]))
        idxs1 = np.asarray(choices(all_idxs, cum_weights=w, k = n))
        idxs2 = np.asarray(list(set(all_idxs) - set(idxs1)))
        return idxs1, idxs2

    def fit(self, _data):
        data = _data[:,:-1]

        mean = np.mean(data, axis = 0)
        dataNorm = data - mean
        u, s, v = np.linalg.svd(dataNorm, full_matrices=1, compute_uv=1)

        # return the estimated line in the form: ax + by + c = 0
        a = v[0,1]
        b = v[1,1]
        c = -(mean[0] * a + mean[1] * b)
        if not a and not b and not c:
            self.logger.critical("Incorrect SVD decomposition, all values are null!")
            self.logger.critical("data shape: %s", data.shape)
            self.logger.critical("v: %s", v)
            raise ValueError("Aborting execution, need to debug")
        return (a, b , c)

    def get_error(self, data, model):
        a, b, c = model

        poses = np.vstack((
                            data[:, 0].tolist(),
                            data[:, 1].tolist(),
                            [np.ones(data.shape[0])]
                        )).T
        err = np.fabs(np.dot(poses, np.float32([a, b, c])))
        return err

class ransac:
    """A class thresholding the image """
    def __init__(self, conf, loggerName = None):
        self.logger = logging.getLogger(loggerName)
        self.conf = conf

    def compute(self, imgIn, _lines):
        if 'method' in self.conf.keys():
            method = self.conf['method']
        else:
            self.logger.error('No ransac method solution specified, fallback to opencv2 svd')
            method = 'svd'
        width = int(self.conf['window']) if 'window' in self.conf.keys() else RANSAC_LINE_DEFAULT_BOUNDING_BOX_WIDTH;
        n = int(self.conf['samples']) if 'samples' in self.conf.keys() else RANSAC_LINE_DEFAULT_SAMPLES
        k = int(self.conf['iterations']) if 'iterations' in self.conf.keys() else RANSAC_LINE_DEFAULT_ITERATIONS
        t = float(self.conf['threshold']) if 'threshold' in self.conf.keys() else RANSAC_LINE_DEFAULT_THRESHOLD
        d = int(self.conf['good']) if 'good' in self.conf.keys() else RANSAC_LINE_DEFAULT_GOOD
        debug = (self.conf['debug'] == '1') if 'debug' in self.conf.keys() else False

        self.logger.debug('Ransac using %s method, (w,n,k,t,d): (%s, %s, %s, %s, %s)', method, width, n, k, t, d)

        if method == 'svd':
            ransacMethod = ransacModel.SVD
        else:
            self.logger.error('Only SVD method supported')
            ransacMethod = ransacModel.SVD

        ransac_lines = []
        for _line in _lines:
            # Get bounding box coordinates
            box = [int(i[j]) for i in _line.getBoundingBox(int(width)) for j in range(len(i))]
            box = list(zip(box[0::2], box[1::2]))
            # Extract subimage : ROI of image acccording to box
            subImage = np.array(np.zeros(imgIn.shape), dtype=imgIn.dtype)
            subImage[box[0][0]:box[1][0], box[0][1]:box[1][1]] = imgIn[box[0][0]:box[1][0], box[0][1]:box[1][1]]
            # Get non zero point only
            # Using numpy, but opencv2.findNonZero(img) could be used ( would required uint8)
            nz = np.nonzero(subImage)
            if not len(nz[0]):
                self.logger.error("Skipping this line - all pixels are nulls in box: %s", box)
                continue
            points = np.empty([len(nz[0]), 3], dtype=np.float32)
            points[:,0:2] = np.transpose(nz).astype(np.float32)
            points[:,2] = subImage[nz]
            # Do Line Ransac estimation on subimage
            model = npSvdModel(self.logger, debug=debug)
            ransac_fit, ransac_data = doRansac(points, model, n, k, t, d, debug = debug, return_all = True, logger=self.logger)
            if ransac_fit is None:
                self.logger.debug("RANSAC did not fount out a line within this box %s", box);
                continue
            self.logger.debug("Line standard equation: ax + by + c = 0: %s", ransac_fit);
            a, b, c = ransac_fit
            if not a and not b and not c:
                raise ValueError("All lines parameters (a, b, c) are null!")
            # Compute it as sympy Line wants
            theta = atan2(b, a)
            r = -c/sqrt(a*a + b*b)
            if r < 0:
                r = abs(r)
                theta += pi
                if theta > pi:
                    theta -= 2 * pi
            p1 = Point((r * cos(theta)), (r * sin(theta)))
            slope = tan(pi/2 - theta)
            roi = Polygon(Point(0, 0), Point(imgIn.shape[1] - 1, 0), Point(imgIn.shape[1] - 1, imgIn.shape[0] - 1), Point(0, imgIn.shape[0] - 1))
            l = Line(p1, slope = slope)
            pts = roi.intersection(l)
            ransac_lines.append(line(*pts, imageBox = [(0, 0), (imgIn.shape[1] - 1, imgIn.shape[0] - 1)]))

        return ransac_lines
