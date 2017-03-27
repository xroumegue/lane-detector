import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os

import numpy as np;
import cv2;
import math;

DETECTOR_LOGGER_NAME = "Lanes Detector"

def interpolation(array,x,y):
    s = array.shape
    i = math.floor(x)
    j = math.floor(y)
    t = x-i
    u = y-j
    u1 = 1.0-u
    t1 = 1.0-t
    if j==s[0]-1:
        if i==s[1]-1:
            return array[j][i]
        return t*array[j][i]+t1*array[j][i+1]
    if i==s[1]-1:
        return u*array[j][i]+u1*array[j+1][i]
    return t1*u1*array[j][i]+t*u1*array[j+1][i]+t*u*array[j+1][i+1]+t1*u*array[j][i+1]

class ipm:
    def __init__(self, conf):
       self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
       self.conf = conf
       self.vp = self._getVanishingPoint()
       self.roi = self._getROI()

    def _getVanishingPoint(self):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']

        vp = np.float32([[ s2/c1, c2/c1, 0 ]]).transpose()

        Tyaw = np.float32([
                            [ c2,   -s2,    0   ],
                            [ s2,   c2,     0   ],
                            [ 0,    0,      1   ],
                        ]
                )

        Tpitch = np.float32([
                                [ 1,    0,      0   ],
                                [ 0,    -s1,    -c1 ],
                                [ 0,    c1,     -s1 ],
                        ]
                )


        T1 = np.float32([
                            [ fu,   0,      cu  ],
                            [ 0,    fv,     cv  ],
                            [ 0,    0,      1   ],
                    ]
                )


        T = np.dot(Tpitch, Tyaw)
        T = T1.dot(T)
        vp = T.dot(vp)
        self.logger.debug('Vanishing point coordinates: (%.3f, %.3f)', vp[0], vp[1])

        return vp

    def Tw2i(self, _in):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        _in = np.insert(_in, _in.shape[0], [[-h],[1]], axis=0)
        Mw2i = np.float32(
                [
                    [ fu*c2 + cu*c1*s2,	cu*c1*c2 - s2*fu, -cu*s1, 0 ],
                    [ s2*(cv*c1 - fv*s1), c2*(cv*c1 - fv*s1), -fv*c1 - cv*s1, 0 ],
                    [ c1*s2, c1*c2, -s1, 0 ],
                    [ c1*s2, c1*c2, -s1, 0 ],
                ]
            )

        i = np.dot(Mw2i, _in)
        i *= np.ones(i.shape)/i[i.shape[0]-1,:]

        return i[[0,1],:]

    def _getROI(self):
        vp = self.vp

        ipmTop = self.conf['ipmTop']
        ipmBottom = self.conf['ipmBottom']
        ipmRight = self.conf['ipmRight']
        ipmLeft = self.conf['ipmLeft']
        ipmWidth = self.conf['ipmWidth']
        ipmHeight = self.conf['ipmHeight']

        iROI = np.float32(
                [
                    [ vp[0], ipmTop],
                    [ ipmRight, ipmTop ],
                    [ ipmLeft, ipmTop ],
                    [ vp[0], ipmBottom ],
                ]
            ).transpose()

        wROI = self.Ti2w(iROI)

        _ROI = {
                'x' : {
                        'min' : np.amin(wROI[0, :]),
                        'max' : np.amax(wROI[0, :]),
                        'scale' : (np.amax(wROI[0, :]) - np.amin(wROI[0, :]))/ipmWidth,
                    },
                'y' : {
                        'min' : np.amin(wROI[1, :]),
                        'max' : np.amax(wROI[1, :]),
                        'scale' : (np.amax(wROI[1, :]) - np.amin(wROI[1, :]))/ipmHeight,
                    }
                }

        self.logger.debug('wROI: (%.2f, %.2f)(%.2f,%.2f)', _ROI['x']['min'], _ROI['y']['min'], _ROI['x']['max'], _ROI['y']['max'])
        return _ROI


    def Ti2w(self, _in):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        _in = np.insert(_in, _in.shape[0], [[1],[1]], axis=0)
        Mi2w = np.float32(
                [
                    [ -c2/fu, s1*s2/fv,	(cu*c2/fu) - (cv*s1*s2/fv) - (c1*s2), 0 ],
                    [ s2/fu, s1*c1/fv, (-cu*s2/fu) - (cv*s1*c2/fv) - (c1*c2), 0 ],
                    [ 0, c1/fv,	(-cv*c1/fv) +s1, 0 ],
                    [ 0, -c1/(h*fv), (cv*c1/(h*fv)) - (s1/h), 0 ],
                ]
            )

        w = np.dot(Mi2w, _in)
        w *= np.ones(w.shape)/w[w.shape[0]-1,:]

        return w[[0,1],:]

    def getVanishingPoint(self):
        return self.vp

    def getROI(self):
        return self.roi

    def load(self, _file):
        self.img = cv2.imread(_file, cv2.IMREAD_GRAYSCALE)

    def compute(self):
        width = self.conf['ipmWidth']
        height = self.conf['ipmHeight']
        right = self.conf['ipmRight']
        left = self.conf['ipmLeft']
        top = self.conf['ipmTop']
        bottom = self.conf['ipmBottom']

        out = np.zeros([height, width], dtype= np.uint8)
        self.out = out

        wOut = np.zeros([height,width, 2], dtype=np.float32)
        wOut[:,:,0] =  np.repeat(np.linspace(self.roi['x']['min'], self.roi['x']['max'], num=width, dtype= np.float32)[np.newaxis,:], height, 0)
        wOut[:,:,1] =  np.repeat(np.linspace(self.roi['y']['max'], self.roi['y']['min'], num=height, dtype= np.float32)[:,np.newaxis], width, 1)

        wPos = np.reshape(wOut, (width*height, 2));

        iPosVector = self.Tw2i(wPos.transpose()).transpose()
        iPos = np.reshape(iPosVector, (height, width, 2))

        for x in range(width):
            for y in range(height):
                xPos = iPos[y, x, 0]
                yPos = iPos[y, x, 1]

                if left <= xPos <= right and top <= yPos <= bottom:
                    if self.conf['ipmInterpolation'] == 0:
                        #bilinear interpolation
                        out[y, x] = interpolation(self.img, xPos, yPos)
                    else:
                        # Nearest Neighbour
                        out[y, x] = self.img[yPos.astype(int), xPos.astype(int)]
                else:
                   out[y, x] =  0

        return self.out

class filter:
    """A class filtering road image"""
    def __init__(self, conf):
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
        self.conf = conf

    def customFilter(self, imgIn, lineType= 'vertical'):
        kernel = {
                    'vertical' : {
                        #High pass filter (edge)
                        'x' : np.float32(
                                            [
                                                1.000000e-16, 1.280000e-14, 7.696000e-13,
                                                2.886400e-11, 7.562360e-10, 1.468714e-08,
                                                2.189405e-07, 2.558828e-06, 2.374101e-05,
                                                1.759328e-04, 1.042202e-03, 4.915650e-03,
                                                1.829620e-02, 5.297748e-02, 1.169560e-01,
                                                1.918578e-01, 2.275044e-01, 1.918578e-01,
                                                1.169560e-01, 5.297748e-02, 1.829620e-02,
                                                4.915650e-03, 1.042202e-03, 1.759328e-04,
                                                2.374101e-05, 2.558828e-06, 2.189405e-07,
                                                1.468714e-08, 7.562360e-10, 2.886400e-11,
                                                7.696000e-13, 1.280000e-14, 1.000000e-16
                                            ]
                                            )[:, np.newaxis],
                        # Low pass Filter (smooth)
                        'y' : np.float32(
                                            [
                                                -1.000000e-03, -2.200000e-02, -1.480000e-01,
                                                -1.940000e-01, 7.300000e-01, -1.940000e-01,
                                                -1.480000e-01, -2.200000e-02, -1.000000e-03
                                            ]
                                            )[np.newaxis, :]
                        },
                    'horizontal' : {
                        # High pass filter (edge)
                        'y' : np.float32(
                                            [
                                                -2.384186e-07, -4.768372e-06, -4.482269e-05,
                                                -2.622604e-04, -1.064777e-03, -3.157616e-03,
                                                -6.976128e-03, -1.136112e-02, -1.270652e-02,
                                                -6.776810e-03, 6.776810e-03, 2.156258e-02,
                                                2.803135e-02, 2.156258e-02, 6.776810e-03,
                                                -6.776810e-03, -1.270652e-02, -1.136112e-02,
                                                -6.976128e-03, -3.157616e-03, -1.064777e-03,
                                                -2.622604e-04, -4.482269e-05, -4.768372e-06,
                                                -2.384186e-07
                                            ]
                                            )[:, np.newaxis],
                        # Low pass Filter (smooth)
                        'x' : np.float32(
                                            [
                                                2.384186e-07, 5.245209e-06, 5.507469e-05,
                                                3.671646e-04, 1.744032e-03, 6.278515e-03,
                                                1.778913e-02, 4.066086e-02, 7.623911e-02,
                                                1.185942e-01, 1.541724e-01, 1.681881e-01,
                                                1.541724e-01, 1.185942e-01, 7.623911e-02,
                                                4.066086e-02, 1.778913e-02, 6.278515e-03,
                                                1.744032e-03, 3.671646e-04, 5.507469e-05,
                                                5.245209e-06, 2.384186e-07
                                            ]
                                            )[np.newaxis, :]
                        }
                }
        # filter over the x
        imgNoDC = cv2.subtract(imgIn, cv2.mean(imgIn))
        imgOutx = cv2.filter2D(imgNoDC, -1, kernel[lineType]['x'])
        imgOuty = cv2.filter2D(imgOutx, -1, kernel[lineType]['y'])

        return imgOuty

    def compute(self, imgIn):
        if self.conf['method'] == 'custom':
            imgOut = self.customFilter(imgIn)
        elif self.conf['method'] == 'bilateral':
            imgOut = cv2.bilateralFilter(outImg, 15, 100, 30)

        return imgOut

class threshold:
    """A class thresholding the image """
    def __init__(self, conf):
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
        self.conf = conf

    def compute(self, imgIn):

        if 'method' in self.conf.keys():
            method = self.conf['method']
        else:
            self.logger.error('No threshold method solution, fallback to binary')
            method = 'binary'

        maxValue = np.iinfo(imgIn.dtype).max
        thresholdValue =  np.percentile(imgIn, self.conf['value'])
        self.logger.debug('Percentile value: %.2f, threshold value: %.2f',
                                            thresholdValue, self.conf['value'])
        if method == 'binary':
            cv2Method = cv2.THRESH_BINARY
        elif method == 'tozero':
            cv2Method = cv2.THRESH_TOZERO

        ret, imgOut = cv2.threshold(imgIn, thresholdValue, maxValue, cv2Method)

        return imgOut

class line:
    """ A class describing a line(https://en.wikipedia.org/wiki/Line_(geometry)) in a image """
    def __init__(self, ptsPolar, score = None, box = None):
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
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
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)
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

class laneDetector:
    """A class detecting lanes on road picture"""

    def __init__(self):
        FORMAT = '%(asctime)-15s-%(levelname)-5s-%(funcName)-8s-%(lineno)-4s-%(message)s'
        self.config = configparser.ConfigParser(inline_comment_prefixes=('#'))
        # create logger
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(DETECTOR_LOGGER_NAME)

    def readConf(self, _file):
        if os.path.isfile(_file):
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
        conf['box'] = [(0, 0), (img.shape[1] - 1, img.shape[0] - 1)]

        self.lines = lines(conf)
        return self.lines.compute(img)

    def getIPM(self, _file):
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
        myIpm.load(_file)
        outImg = myIpm.compute()
        return outImg

def parse_cmdline(parser):
	parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
	parser.add_argument('-i', '--image', help='Image file')
	parser.add_argument('-c', '--camera', default='CameraInfo.conf', help='Camera configuration file')
	parser.add_argument('-l', '--lanes', default='Lanes.conf', help='Lane configuration file')

	return parser.parse_args()

def main():
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)

    detector = laneDetector()

    detector.logger.setLevel(args.verbose)

    if not args.image or not os.path.isfile(args.image):
        parser.print_help()
        return

    detector.logger.debug('Using %s as image', args.image)
    detector.logger.debug('Using %s as camera configuration file', args.camera)
    detector.logger.debug('Using %s as lanes configuration file', args.lanes)

    detector.readConf(args.camera)
    detector.readConf(args.lanes)

    # IPM
    outImg = detector.getIPM(args.image)
    cv2.imwrite("IPM-out.jpg", outImg)
    cv2.imshow('IPM', outImg);

    # Filter out "noise"
    outImgFiltered = detector.filter(outImg)
    cv2.imshow('IPM Filtered', outImgFiltered);

    # Threshold
    outImgThresholded = detector.threshold(outImg)
    cv2.imshow('IPM Thresholded', outImgThresholded);

    # Detecting lines
    myLines = detector.lines(outImgThresholded)
    for _line in myLines:
        __line = _line.getCartesian()
        cv2.line(outImgThresholded, tuple(round(float(_)) for _ in __line[0]), tuple(round(float(_)) for _ in __line[1]), (255, 0, 0), 1)

    cv2.imshow('IPM vertical lines', outImgThresholded);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

if __name__ == '__main__':
    main()
