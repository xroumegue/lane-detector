from argparse import ArgumentParser, FileType, Action, Namespace
from os import listdir
from os.path import isfile, dirname, realpath, join
from sys import path
import logging
import cv2

customLibPath=join(dirname(realpath(__file__)),'../lib/python')
for _dir in listdir(customLibPath):
    path.append(dirname(realpath(join(customLibPath, _dir))))


import laneDetector

def parse_cmdline(parser):
	parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
	parser.add_argument('-i', '--image', help='Image file')
	parser.add_argument('-c', '--camera', default='CameraInfo.conf', help='Camera configuration file')
	parser.add_argument('-l', '--lanes', default='Lanes.conf', help='Lane configuration file')

	return parser.parse_args()

def main():
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)

    detector = laneDetector.laneDetector()

    detector.logger.setLevel(args.verbose)

    if not args.image or not isfile(args.image):
        parser.print_help()
        return

    detector.logger.debug('Using %s as image', args.image)
    detector.logger.debug('Using %s as camera configuration file', args.camera)
    detector.logger.debug('Using %s as lanes configuration file', args.lanes)

    detector.readConf(args.camera)
    detector.readConf(args.lanes)

    # Load Image
    detector.load(args.image)
    detector.logger.debug('Original Image energy: %f \n', detector.getEnergy(detector.scaleImage))
    detector.showImage('original', detector.rawImage)

    # IPM
    outImg = detector.getIPM(useRaw = False)
    # IPM debug
#    cv2.imwrite("IPM-out.jpg", outImg)
    detector.showImage('IPM', outImg)
    detector.logger.debug('IPM Image energy: %f', detector.getEnergy(outImg))

    # Filter out "noise"
    outImgFiltered = detector.filter(outImg)
    # Filter debug
    detector.logger.debug('Filtered Image energy: %f', detector.getEnergy(outImgFiltered))
    detector.showImage('IPM Filtered', outImgFiltered);

    # Threshold
    outImgThresholded = detector.threshold(outImgFiltered)
    # Threshold debug
    detector.logger.debug('Thresholded Image energy: %f', detector.getEnergy(outImgThresholded))
    detector.showImage('IPM Thresholded', outImgThresholded);

    # Detecting lines
    myLines = detector.lines(outImgThresholded)
    # Detecting lines debug
    detector.showLines('IPM vertical lines', outImgThresholded, myLines)

    # Ransac
    ransacLines = detector.ransac(outImgThresholded, myLines)
    # RAnsac lines debug
    detector.showLines('IPM Ransac lines', outImgThresholded, ransacLines)

    cv2.waitKey(0);
    cv2.destroyAllWindows();

if __name__ == '__main__':
    main()
