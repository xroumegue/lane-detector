import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os
import numpy as np;
import cv2;
import math;
import laneDetector
from ransac import ransacModel, getModel, doRansac
from laneDetector.lines import line

RANSAC_LINE_DEFAULT_BOUNDING_BOX_WIDTH = 20
RANSAC_LINE_DEFAULT_SAMPLES = 5
RANSAC_LINE_DEFAULT_ITERATIONS = 50
RANSAC_LINE_DEFAULT_THRESHOLD = 0.3
RANSAC_LINE_DEFAULT_GOOD = 10

class npSvdModel:
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        mean = np.mean(data, axis = 0)
        dataNorm = data - mean
        u, s, v = np.linalg.svd(dataNorm, full_matrices=1, compute_uv=1)

        # return the estimated line in the form: ax + by + c = 0
        a = v[0,1]
        b = v[1,1]
        c = -(mean[0] * a + mean[1] * b)
#        print(a, b, c)
        return (a, b , c)

    def get_error(self, data, model):
        a, b, c = model

        poses = np.vstack((
                            [data[:,i] for i in self.input_columns],
                            [data[:,i] for i in self.output_columns],
                            [np.ones(data.shape[0])]
                        )).T
        err = np.fabs(np.dot(poses, np.float32([a, b, c])))
        return err

class ransac:
    """A class thresholding the image """
    def __init__(self, conf):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
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
            points = np.transpose(np.nonzero(subImage)).astype(np.float)
            # Do Line Ransac estimation on subimage
            model = npSvdModel([0], [1], debug=debug)
            ransac_fit, ransac_data = doRansac(points, model, n, k, t, d, debug = debug, return_all = True, logger=self.logger)
            self.logger.debug("Line standard equation: ax + by + c = 0: %s", ransac_fit);
            ransac_lines.append(line([ransac_fit], imageBox = [(0, 0), (imgIn.shape[1] - 1, imgIn.shape[0] - 1)]))

        return ransac_lines
