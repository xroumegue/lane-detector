import logging
import configparser
import numpy as np
import cv2
from os.path import isfile
from laneDetector import *
from laneDetector.ipm import ipm
from laneDetector.filters import filter
from laneDetector.lines import lines
from laneDetector.threshold import threshold
from laneDetector.ransac import ransac

__all__ = ['DETECTOR_LOGGER_NAME', 'ipm', 'filters', 'threshold', 'lines',
        'laneDetector', 'ransac']

__author__  = "Xavier Roumegue <xroumegue.gmail.com>"
__status__  = "prototyping"
__version__ = "0.0.1"
__date__    = "29 March 2017"

DETECTOR_LOGGER_NAME = "Lanes Detector"

class laneDetector:
    """A class detecting lanes on road picture"""

    def __init__(self):
        FORMAT = '%(asctime)-15s-%(levelname)-5s-%(funcName)-8s-%(lineno)-4s-%(message)s'
        self.config = configparser.ConfigParser(inline_comment_prefixes=('#'))
        # create logger
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
        self.rawImage = None
        self.scaleImage = None

    def scale(self, img):
        if img.dtype == np.uint8:
            _img = np.float32(img / np.iinfo(img.dtype).max)
        else:
            if self.rawImage is not None:
                _scale = np.iinfo(self.rawImage.dtype).max
            else:
                _scale = 255
            _img = cv2.convertScaleAbs(img, alpha = _scale)

        return _img

    def load(self, _file):
        self.rawImage = cv2.imread(_file, cv2.IMREAD_GRAYSCALE)
        self.scaleImage = self.scale(self.rawImage)

    def showImage(self, title, img):
        if img.dtype == np.uint8:
            _img = img
        else:
            _img = self.scale(img)

        cv2.imshow(title, _img)

    def showLines(self, title, _img, _lines):
        img = np.array(np.empty(_img.shape, dtype=_img.dtype))
        np.copyto(img, _img)
        for _line in _lines:
            __line = _line.getCartesian()
            cv2.line(img, tuple(round(float(_)) for _ in __line[0]), tuple(round(float(_)) for _ in __line[1]), (255, 0, 0), 1)

        self.showImage(title, img);

    def readConf(self, _file):
        if isfile(_file):
            self.config.read(_file)
        else:
            self.logging.error("Config file %s does not exist !", _file)

    def filter(self, img):
        conf = {}
        if not self.config.has_section('filter'):
            self.logger.error('No filter configuration found !')
            return

        for (k, v) in self.config.items('filter'):
            conf[k] = v

        self.filter = filter(conf)
        imgOut = self.filter.compute(img)
        return imgOut

    def threshold(self, img):
        conf = {}
        if not self.config.has_section('threshold'):
            self.logger.error('No threshold configuration found !')
            return

        for (k, v) in self.config.items('threshold'):
            conf[k] = v
        conf['value'] = self.config.getfloat('threshold', 'value')

        self.threshold = threshold(conf)
        imgOut = self.threshold.compute(img)
        return imgOut

    def lines(self, img):
        conf = {}
        if not self.config.has_section('lines'):
            self.logger.error('No lines configuration found !')
            return

        for (k, v) in self.config.items('lines'):
            conf[k] = v

        conf['threshold'] = self.config.getfloat('lines', 'threshold')
        conf['minDistance'] = self.config.getfloat('lines', 'minDistance')
        conf['imageBox'] = [(0, 0), (img.shape[1] - 1, img.shape[0] - 1)]

        self.lines = lines(conf)
        return self.lines.compute(img)

    def ransac(self, img, lines):
        conf = {}
        if not self.config.has_section('ransac'):
            self.logger.error('No ransac configuration found')
            return

        for (k, v) in self.config.items('ransac'):
            conf[k] = v

        self.ransac = ransac(conf)
        ransacLines = self.ransac.compute(img, lines)
        for _line in ransacLines:
            _line.setImageBox([(0, 0), (img.shape[1] - 1, img.shape[0] - 1)])

        return ransacLines

    def getIPM(self, useRaw=False):
        conf = {}
        conf['yaw'] = self.config.getfloat('camera', 'yaw') * np.pi / 180
        conf['pitch'] = self.config.getfloat('camera', 'pitch') * np.pi / 180
        conf['c1'] = np.cos(conf['yaw'])
        conf['s1'] = np.sin(conf['pitch'])
        conf['c2'] = np.cos(conf['yaw'])
        conf['s2'] = np.sin(conf['yaw'])
        conf['fu'] = self.config.getfloat('camera', 'focalLengthX')
        conf['fv'] = self.config.getfloat('camera', 'focalLengthY')
        conf['cu'] = self.config.getfloat('camera', 'opticalCenterX')
        conf['cv'] = self.config.getfloat('camera', 'opticalCenterY')
        conf['h'] = self.config.getfloat('camera', 'cameraHeight')

        conf['ipmWidth']  = self.config.getint('ipm', 'ipmWidth')
        conf['ipmHeight']  = self.config.getint('ipm', 'ipmHeight')
        conf['ipmTop']  = self.config.getint('ipm', 'ipmTop')
        conf['ipmBottom'] = self.config.getint('ipm', 'ipmBottom')
        conf['ipmLeft'] = self.config.getint('ipm', 'ipmLeft')
        conf['ipmRight'] = self.config.getint('ipm', 'ipmRight')
        conf['ipmInterpolation'] = self.config.getint('ipm', 'ipmInterpolation')

        myIpm = ipm(conf)

        myIpm.getVanishingPoint()
        self.logger.info('Vanishing point: (%.2f, %.2f)', myIpm.vp[0], myIpm.vp[1])
        myIpm.getROI()
        _img = self.scaleImage if useRaw is False else self.rawImage
        outImg = myIpm.compute(_img)
        return outImg


