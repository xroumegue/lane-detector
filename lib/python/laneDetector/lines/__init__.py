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
    def __init__(self, ptsPolar, score = None, imageBox = None):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
        self.pts = None
        self.r = None
        self.theta = None
        self.origin = None
        self.score = score if score is not None else None
        self.imageBox = imageBox if imageBox is not None else None

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
            if len(ptsPolar[0]) == 2:
                self.r = ptsPolar[0][0]
                self.theta = ptsPolar[0][1]
            elif len(ptsPolar[0]) == 3:
                """ ax + by + c = 0"""
                a, b, c = ptsPolar[0]
                inv_angle = math.sqrt(a*a + b*b)
                if c > 0:
                    inv_angle = -inv_angle
                self.r = math.fabs(c / inv_angle)
                x_angle = math.acos(a/inv_angle)
                y_angle = math.asin(b/inv_angle)
                z_angle = math.atan2(b, a)
                self.theta = z_angle
        elif len(ptsPolar) == 2:
            self.pts = ptsPolar

        self.logger.debug("Line created: r(%s) theta(%s), pts(%s), Score(%s), ImageBox(%s)", self.r, self.theta, self.pts, self.score, self.imageBox)

    def getBoundingBox(self, width):
        pts = self.getCartesian()
        # Vertical line
        if pts[0][0] == pts[1][0]:
            return [(max(self.imageBox[0][0], pts[0][0] - width), pts[0][1]),
                    (min(self.imageBox[1][0], pts[1][0] + width), pts[1][1])]
        elif pts[0][1] == pts[1][1]:
            return [(pts[0][0], max(self.imageBox[0][1], pts[0][1] - width)),
                    (pts[1][0], min(self.imageBox[1][1], pts[1][1] + width))]
        else:
        # TODO
            self.logger.error("General case of bounding box: untested, returning 4 points tuples")
            a = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            l_a = math.sqrt(a[0]*a[0] + a[1]*a[1])
            by = a[0] * width/l_a
            bx = math.sqrt((width * width) - (by*by))
            return [(min(pts[0][0], pts[1][0]) - bx,min(pts[0][1],pts[1][1]) - by),
                    (min(pts[0][0], pts[1][0]) + bx,min(pts[0][1],pts[1][1]) + by),
                    (max(pts[0][0], pts[1][0]) - bx,max(pts[0][1],pts[1][1]) - by),
                    (max(pts[0][0], pts[1][0]) + bx,max(pts[0][1],pts[1][1]) + by)]



    def setImageBox(self, box):
        self.imageBox = box

    def getImageBox(self):
        return self.imageBox

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

    def _computeBox(self):
        minX = min(self.pts[0][0], self.pts[1][0])
        maxX = max(self.pts[0][0], self.pts[1][0])
        minY = min(self.pts[0][1], self.pts[1][1])
        maxY = max(self.pts[0][1], self.pts[1][1])

        if minX != maxX and minY != maxY:
            pass
        elif minX == maxX:
            if minX != self.getImageBox()[0][0]:
                minX -= 1
            else:
                maxX += 1
        else:
            if minY != self.getImageBox()[0][1]:
                minY -= 1
            else:
                maxY += 1

        self.box = [(minX, minY), (maxX, maxY)]

    def _getCartesian(self):
        if self.pts is not None:
            return self.pts

        if self.r is None or self.theta is None:
            self.logger.error("A line must have either polar or cartesian coordinate")

        if self.theta == math.pi or self.theta == 0:
            # vertical lines
            pts = [(self.r, self.imageBox[0][1]), (self.r, self.imageBox[1][1])]
        elif self.theta == math.pi/2 or self.theta == -math.pi/2:
            # Horizontal lines
            pts = [(self.imageBox[0][0], self.r), (self.imageBox[1][0], self.r)]
        else:
            # General case
            # r = x * cos(theta) + y * sin(theta)
            o = self.getOrigin()
            y = sorted([self.imageBox[0][1], self.imageBox[1][1]])
            x = sorted([self.imageBox[0][0], self.imageBox[1][0]])

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
        self._computeBox()

        return self.pts

    def getCartesian(self, imageBox = None):
        if self.pts is None:
            if imageBox is None and self.imageBox is None:
                self.logger.error("A box must be specified")
                return
            if imageBox is not None:
                self.setImageBox(imageBox)
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
                gLines[i] = line([(p, t)], score, imageBox = self.conf['imageBox'])
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

            lines.append(line(myLine, imgVector[index], imageBox = self.conf['imageBox']))

        for _line in lines:
            self.logger.debug("%s lines detected: %s", lineType, _line.getCartesian())

        return lines

    def compute(self, imgIn, lineType = 'vertical'):
        if self.conf['method'] == 'custom':
            self.rawLines = self.customCompute(imgIn, lineType)

        self.groupLines = self.group(self.rawLines)
        return self.groupLines


