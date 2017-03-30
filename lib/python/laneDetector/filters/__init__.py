import logging

import numpy as np;
import cv2;
import math;
import laneDetector

class filter:
    """A class filtering road image"""
    def __init__(self, conf):
        self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
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

