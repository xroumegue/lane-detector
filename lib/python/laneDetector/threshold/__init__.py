import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os
import numpy as np;
import cv2;
import math;
import laneDetector

class threshold:
    """A class thresholding the image """
    def __init__(self, conf):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
        self.conf = conf

    def compute(self, imgIn):

        if 'method' in self.conf.keys():
            method = self.conf['method']
        else:
            self.logger.error('No threshold method solution, fallback to binary')
            method = 'binary'

        if np.issubdtype(imgIn.dtype, int):
            maxValue = np.iinfo(imgIn.dtype).max
        else:
            maxValue = 1.0

        thresholdValue =  np.percentile(imgIn, self.conf['value'])

        if method == 'binary':
            cv2Method = cv2.THRESH_BINARY
        elif method == 'tozero':
            cv2Method = cv2.THRESH_TOZERO

        self.logger.debug('Percentile value: %.2f, threshold value: %.2f, method: %s',
                                            thresholdValue, self.conf['value'], method)

        ret, imgOut = cv2.threshold(imgIn, thresholdValue, maxValue, cv2Method)

        return imgOut

