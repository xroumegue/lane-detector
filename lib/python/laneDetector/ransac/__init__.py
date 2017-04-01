import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os
import numpy as np;
import cv2;
import math;
import laneDetector
from ransac import ransacModel, getModel, doRansac

RANSAC_LINE_DEFAULT_BOUNDING_BOX_WIDTH = 20
RANSAC_LINE_DEFAULT_SAMPLES = 5
RANSAC_LINE_DEFAULT_ITERATIONS = 50
RANSAC_LINE_DEFAULT_THRESHOLD = 0.3
RANSAC_LINE_DEFAULT_GOOD = 10

class ransac:
    """A class thresholding the image """
    def __init__(self, conf):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
        self.conf = conf

    def compute(self, imgIn, lines):
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
        elif method == 'lstsq':
            ransacMethod = ransacModel.LSTSQ
        else:
            ransacMethod = ransacModel.SVD


        for _line in lines:
            # Get bounding box coordinates
            box = [int(i[j]) for i in _line.getBoundingBox(int(width)) for j in range(len(i))]
            box = list(zip(box[0::2], box[1::2]))
            # Extract subimage : ROI of image acccording to box
            subImage = imgIn[box[0][0]:box[1][0], box[0][1]:box[1][1]]
            # Get non zero point only
            # Using numpy, but opencv2.findNonZero(img) could be used ( would required uint8)
            points = np.transpose(np.nonzero(subImage)).astype(np.float)
            print(points.shape)
            x = points[:,0]
            y = points[:,1]
            # Do Line Ransac estimation on subimage
            model = getModel(ransacMethod)([0], [1], debug=debug)
            ransac_fit, ransac_data = doRansac(points, model, n, k, t, d, debug = debug, return_all = True, logger=self.logger)



        return imgIn
