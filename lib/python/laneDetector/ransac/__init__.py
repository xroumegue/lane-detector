import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os
import numpy as np;
import cv2;
import math;
import laneDetector
import ransac

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

        self.logger.debug('Ransac using %s method', method)


        return imgOut
