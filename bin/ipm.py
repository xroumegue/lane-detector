#! /usr/bin/env python3

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
from PIL import Image
import logging
import numpy as np

from argparse import ArgumentParser, FileType, Action, Namespace

FORMAT = '%(asctime)-15s-%(levelname)-5s-%(funcName)-8s-%(lineno)-4s-%(message)s'

ESCAPE = '\033'

def parse_cmdline(parser):
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
    parser.add_argument('-i', '--image', help='Image file')

    return parser.parse_args()

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global window
    # If escape is pressed, kill everything.
    if args[0].decode() == ESCAPE:
        sys.exit()

class Texture( object ):
        """Texture either loaded from a file."""
        def __init__( self ):
                self.xSize, self.ySize = 0, 0
                self.rawRefence = None

class FileTexture( Texture ):
        """Texture loaded from a file."""
        def __init__( self, fileName ):
                im = Image.open(fileName)
                self.xSize = im.size[0]
                self.ySize = im.size[1]
                self.rawReference=np.array(list(im.getdata()),np.uint8)

def display(  ):
        """Glut display function."""
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glColor3f( 1, 1, 1 )
        glBegin( GL_QUADS )
        glTexCoord2f( 0, 1 )
        glVertex3f( -0.5, -0.5, 0 )
        glTexCoord2f( 0, 0 )
        glVertex3f( -0.5, +0.5, 0 )
        glTexCoord2f( 1, 0 )
        glVertex3f( 0.5, +0.5, 0 )
        glTexCoord2f( 1, 1 )
        glVertex3f( 0.5, -0.5, 0 )
        glEnd(  )
        glutSwapBuffers (  )

def init( fileName ):
        """Glut init function."""
        try:
                texture = FileTexture( fileName )
        except:
                print('could not open ', fileName, '; using random texture')
                texture = RandomTexture( 256, 256 )
        glClearColor ( 0, 0, 0, 0 )
        glShadeModel( GL_SMOOTH )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR )
        glTexImage2D( GL_TEXTURE_2D, 0, 3, texture.xSize, texture.ySize, 0,
                                 GL_RGB, GL_UNSIGNED_BYTE, texture.rawReference )
        glEnable( GL_TEXTURE_2D )


def main():
    logging.basicConfig(format=FORMAT)
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)
    log = logging.getLogger("ipm openGL")
    log.setLevel(args.verbose)
    log.info("OpenGL acceleration to compute an IPM")

    glutInit( sys.argv )
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB )
    glutInitWindowSize( 250, 250 )
    glutInitWindowPosition( 100, 100 )
    glutCreateWindow( sys.argv[0] )
    init(args.image)

    # Register the function called when the keyboard is pressed.
    glutKeyboardFunc(keyPressed)

    glutDisplayFunc( display )
    glutMainLoop(  )

print("Hit ESC key to quit.")
main()
