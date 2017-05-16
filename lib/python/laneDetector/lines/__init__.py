import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os

from sympy import Point
from sympy.geometry import Line, intersection, Polygon

import numpy as np;
import cv2;
import math;

class line(Line):
    def __new__(self, *args, **kwargs):
        loggerName = None
        self.imageBox = None
        score = None
        if kwargs is not None:
            __keys = kwargs.keys()
            if 'loggerName' in __keys:
                loggerName = kwargs['loggerName']
            if 'imageBox' in __keys:
                box = kwargs['imageBox']
                self.imageBox = Polygon(Point(box[0][0], box[0][1]), Point(box[1][0], box[0][1]), Point(box[1][0], box[1][1]), Point(box[0][0], box[0][1]))
            if 'score' in __keys:
                score = kwargs['score']

        self.logger = logging.getLogger(loggerName)
        self.score = score
        self.origin = None
        self.r = None
        self.theta = None
        return super().__new__(self, *args, **kwargs)

    def _getPolar(self):
        #vertical line
        if self.p1.y == self.p2.y:
            self.r = abs(self.p1.x)
            self.theta = 0. if self.p1.x >=0 else math.pi;
        #Horizontal line
        elif self.p1.y == self.p2.y:
            self.r = abs(self.p1.y)
            self.theta = math.pi/2 if self.p1.y >=0 else -math.pi/2;
        # General case
        else:
            self.theta = math.atan2((self.p2.x - self.p1.x), (self.p1.y - self.p2.y))
            self.r = r1 = self.p1.x * math.cos(self.theta) + self.p1.y * math.sin(self.theta)
            r2 = self.p2.x * math.cos(self.theta) + self.p2.y * math.sin(self.theta)
            if r1 < 0 or r2 < 0:
                self.theta += math.pi
                if self.theta > math.pi:
                    self.theta -= 2 * math.pi
                self.r = abs(r1)

        return (self.r, self.theta)

    def getPolar(self):
        if self.r is None or self.theta is None:
            self._getPolar()

        return (self.r, self.theta)

    def _getOrigin(self):
        (r, theta) = self.getPolar()
        self.origin = (r * math.cos(theta), r * math.sin(theta))

        return self.origin

    def getOrigin(self):
        if self.origin is None:
            self._getOrigin()

        return self.origin

    def setImageBox(self, box):
        self.imageBox = Polygon(Point(box[0][0], box[0][1]), Point(box[1][0], box[0][1]), Point(box[1][0], box[1][1]), Point(box[0][0], box[0][1]))

    def getImageBox(self):
        return self.imageBox

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score

    def getCartesian(self, imageBox = None):

        (p1, p2) = self.points
        return ((p1.x, p1.y), (p2.x, p2.y))

    def getBoundingBox(self, width):
        (p1, p2) = self.points
        (xmin, ymin, xmax, ymax) = self.imageBox.bounds

        # Vertical line
        if p1.x == p2.x:
            return [(max(xmin, p1.x - width), p1.y),
                    (min(xmax, p2.x + width), p2.y)]
        elif self.p1.y == self.p2.y:
            return [(p1.x, max(ymin, p1.y - width)),
                    (p2.x, min(ymax, p2.y + width))]
        else:
        # TODO
            self.logger.error("General case of bounding box: untested, returning 4 points tuples")
            a = (p2.x - p1.x, p2.y - p1.y)
            l_a = math.sqrt(a[0]*a[0] + a[1]*a[1])
            by = a[0] * width/l_a
            bx = math.sqrt((width * width) - (by*by))
            return [(min(p1.x, p2.x) - bx,min(p1.y, p2.y) - by),
                    (min(p1.x, p2.x) + bx,min(p1.y, p2.y) + by),
                    (max(p1.x, p2.x) - bx,max(p1.y, p2.y) - by),
                    (max(p1.x, p2.x) + bx,max(p1.y, p2.y) + by)]



class lines:
    """A class detecting lines in the image """
    def __init__(self, conf, loggerName = None):
        self.logger = logging.getLogger(loggerName)
        self.conf = conf

    def group(self, _lines):
        gLines = _lines[:]
        minD = self.conf['minDistance']
        i = 0
        while len(gLines) > 1 and i < len(gLines) - 1:
            l1 = gLines[i]
            l2 = gLines[i+1]
            # Assume vertical lanes
            _box = Polygon(
                    Point(l1.p1.x, l1.p1.y), Point(l1.p1.x + minD, l1.p1.y),
                    Point(l1.p2.x + minD, l1.p2.y), Point(l1.p2.x, l1.p2.y)
                )
            if len(_box.intersection(Line(l2.p1, l2.p2))):
                score = (gLines[i].getScore() + gLines[i+1].getScore()) / 2
                p3 = (l1.p1 + l2.p1)/2
                newLine = line(Point(p3.x, l1.p1.y), Point(p3.x, l1.p2.y), score = score, imageBox = self.conf['imageBox'])
                self.logger.debug("Grouping line {} with {} to {}".format(
                                gLines[i].points, gLines[i+1].points, newLine.points))
                gLines[i] = newLine
                del gLines[i+1]
            else:
                i += 1
        return gLines

    def customCompute(self, imgIn, lineType = 'vertical'):
        if 'threshold' in self.conf.keys():
            detectionThreshold = self.conf['threshold']
        else:
            self.logger.error('No threshold value found...using default')
            detectionThreshold = 50 #TODO: Refine me

        if lineType == 'horizontal':
            dimVector = int(1)
        elif lineType == 'vertical':
            dimVector = int(0)

        imgVector = cv2.reduce(imgIn, dimVector, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        if 'smooth' in self.conf.keys() and 'smooth' == 1:
            smoothKernel = np.float32([
                                        0.000003726653172, 0.000040065297393, 0.000335462627903,
                                        0.002187491118183, 0.011108996538242, 0.043936933623407,
                                        0.135335283236613, 0.324652467358350, 0.606530659712633,
                                        0.882496902584595, 1.000000000000000, 0.882496902584595,
                                        0.606530659712633, 0.324652467358350, 0.135335283236613,
                                        0.043936933623407, 0.011108996538242, 0.002187491118183,
                                        0.000335462627903, 0.000040065297393, 0.000003726653172
                                    ]
                    )
            imgVector = cv2.filter2D(imgVector, -1, smoothKernel)

        imgVector = imgVector.flatten()
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(imgVector)
        lines = []
        localMax = []
        for i in range(2, imgVector.size - 2):
            val = imgVector[i]
            if val > imgVector[i-1] \
                    and val > imgVector[i+1]\
                    and val > detectionThreshold:
                localMax.append((i, val))

        if len(localMax) == 0:
            if maxLoc[1] == 0:
                #No line
                self.logger.error("No line detected")
                return []
            localMax.append((maxLoc[1], maxVal))

        # Subpixel
        for (index, val) in localMax:
            x = np.float32([
                            [1, -1, 1],
                            [0,  0, 1],
                            [1,  1, 1]
                    ]
                )
            y = np.float32([imgVector[max(0, index - 1)], val, imgVector[min(len(imgVector) - 1, index)]])
            _, a = cv2.solve(x, y, flags = cv2.DECOMP_SVD)
            indexSub = float((-0.5 * a[1]/a[0]) + index)
            if lineType == 'horizontal':
                myLine = [Point(0.5, indexSub + 0.5), Point(imgIn.shape[1] - 0.5, indexSub + 0.5)]
            elif lineType == 'vertical':
                myLine = [Point(indexSub + 0.5, 0.5), Point(indexSub + 0.5, imgIn.shape[0] - 0.5)]

            lines.append(line(*myLine, score = imgVector[index], imageBox = self.conf['imageBox']))

        for _line in lines:
            self.logger.debug("%s lines detected: %s", lineType, _line.getCartesian())

        return lines

    def compute(self, imgIn, lineType = 'vertical'):
        if self.conf['method'] == 'custom':
            self.rawLines = self.customCompute(imgIn, lineType)

        self.groupLines = self.group(self.rawLines)
        return self.groupLines


