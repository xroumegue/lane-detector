import logging
import configparser
import laneDetector
from argparse import ArgumentParser, FileType, Action, Namespace
import os

import numpy as np;
import cv2;
import math;

class line:
    """ A class describing a line(https://en.wikipedia.org/wiki/Line_(geometry)) in a image """
    def __init__(self, ptsPolar, score = None, box = None):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
        self.pts = None
        self.r = None
        self.theta = None
        self.origin = None
        self.score = score if score is not None else None
        self.box = box if box is not None else None

#       slope = np.tan(self.theta)
#       intercept = self.r / np.sin(self.theta)

        if (not isinstance(ptsPolar, list)):
            self.logger.error("Must be a list")
            return

        if (not all(isinstance(_, tuple) for _ in ptsPolar)):
            self.logger.error("Must have a list of points")
            return

        if (len(ptsPolar) > 2):
            self.logger.error("Must have at most 2 tuples")
            return


        if len(ptsPolar) == 1:
            self.r = ptsPolar[0][0]
            self.theta = ptsPolar[0][1]
        elif len(ptsPolar) == 2:
            self.pts = ptsPolar

    def setBox(self, box):
        self.box = box

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score

    def setPts(self, pts):
        if pts is not None:
            if (not isinstance(pts, list)):
                self.logger.error("Must have a list of points")
                return

            if (not all(isinstance(_, tuple) for _ in pts)):
                self.logger.error("Must have a list of points")
                return

            self.pts = pts
        else:
                self.logger.error("Point list must not be None")

    def setPolar(self, r, theta):
        self.r = r
        self.theta = theta

    def _getOrigin(self):
        (r, theta) = self.getPolar()
        self.origin = (r * math.cos(theta), r * math.sin(theta))

        return self.origin

    def getOrigin(self):
        if self.origin is None:
            self._getOrigin()

        return self.origin

    def _getCartesian(self):
        if self.pts is not None:
            return self.pts

        if self.r is None or self.theta is None:
            self.logger.error("A line must have either polar or cartesian coordinate")

        if self.theta == math.pi or self.theta == 0:
            # vertical lines
            pts = [(self.r, self.box[0][1]), (self.r, self.box[1][1])]
        elif self.theta == math.pi/2 or self.theta == -math.pi/2:
            # Horizontal lines
            pts = [(self.box[0][0], self.r), (self.box[1][0], self.r)]
        else:
            # General case
            # r = x * cos(theta) + y * sin(theta)
            o = self.getOrigin()
            y = [self.box[0][1], self.box[1][1]].sort()
            x = [self.box[0][0], self.box[1][0]].sort()

            if not ((y[1] >= o[1] >= y[0]) and (x[1] >= o[0] >= x[0])):
                self.logger.error('Origin point {} not in box ({} {})'.format(o, x, y))
                return None
            r = self.r
            t = self.theta
            sol = [
                        ((r - y[1] * math.sin(t)) / math.cos(t), y[1]),
                        ((r - y[0] * math.sin(t)) / math.cos(t), y[0]),
                        ((x[0], (r - x[0] * math.cos(t)) / math.sin(t))),
                        ((x[1], (r - x[1] * math.cos(t)) / math.sin(t))),
            ]
            pts = []
            for _ in sol:
                if  x[1] >= _[0] >= x[0]  and y[1] >= _[1] >= y[0]:
                    pts.append(_)

        self.pts = pts
        return self.pts

    def getCartesian(self, box = None):
        if self.pts is None:
            if box is None and self.box is None:
                self.logger.error("A box must be specified")
                return
            if box is not None:
                self.setBox(box)
            self._getCartesian()

        return self.pts

    def getPolar(self):
        if self.r is None or self.theta is None:
            self._getPolar()

        return (self.r, self.theta)

    def _getPolar(self):
        #vertical line
        if self.pts[0][0] == self.pts[1][0]:
            self.r = abs(self.pts[0][0])
            self.theta = 0. if self.pts[0][0] >=0 else math.pi;
        #Horizontal line
        elif self.pts[0][1] == self.pts[1][1]:
            self.r = abs(self.pts[0][1])
            self.theta = math.pi/2 if self.pts[0][1] >=0 else -math.pi/2;
        # General case
        else:
            self.theta = math.atan2((self.pts[1][0] - self.pts[0][0])/(self.pts[0][1] - self.pts[1][1]))
            self.r = r1 = self.pts[0][0] * math.cos(self.theta) + self.pts[0][1] * math.sin(self.theta)
            r2 = self.pts[1][0] * math.cos(self.theta) + self.pts[1][1] * math.sin(self.theta)
            if r1 < 0 or r2 < 0:
                self.theta += math.pi
                if self.theta > math.pi:
                    self.theta -= 2 * math.pi
                self.r = abs(r1)

        return (self.r, self.theta)

class lines:
    """A class detecting lines in the image """
    def __init__(self, conf):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
        self.conf = conf

    def group(self, _lines):
        gLines = _lines[:]
        minD = self.conf['minDistance']
        i = 0
        while len(gLines) > 1 and i < len(gLines) - 1:
            oA = np.asarray(gLines[i].getOrigin())
            oB = np.asarray(gLines[i+1].getOrigin())
            oD = oB - oA
            d = np.linalg.norm(oD)
            if d < minD:
                # TODO/FIXME: Should we determine the new pow 'score weightly'
                o = oA + oD/2
                p = np.linalg.norm(o)
                t = math.atan2(o[1], o[0])
                score = (gLines[i].getScore() + gLines[i+1].getScore()) / 2
                self.logger.debug("Grouping line {} with {} to {}".format(
                                gLines[i].getPolar(), gLines[i+1].getPolar(), (p, t)))
                gLines[i] = line([(p, t)], score, box = self.conf['box'])
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
        if 'smooth' in self.conf.keys():
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
                myLine = [(0.5, indexSub + 0.5), (imgIn.shape[1] - 0.5, indexSub + 0.5)]
            elif lineType == 'vertical':
                myLine = [(indexSub + 0.5, 0.5), (indexSub + 0.5, imgIn.shape[0] - 0.5)]

            lines.append(line(myLine, imgVector[index], box = self.conf['box']))

        for _line in lines:
            self.logger.debug("%s lines detected: %s", lineType, _line.getCartesian())

        return lines

    def compute(self, imgIn, lineType = 'vertical'):
        if self.conf['method'] == 'custom':
            self.rawLines = self.customCompute(imgIn, lineType)

        self.groupLines = self.group(self.rawLines)
        return self.groupLines


