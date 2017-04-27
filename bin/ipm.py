#! /usr/bin/env python3

import sys
try:
	from OpenGL.GL import *
	from OpenGL.GLUT import *
	from OpenGL.GLU import *
	from OpenGL.GL.ARB.shader_objects import *
	from OpenGL.GL.ARB.fragment_shader import *
	from OpenGL.GL.ARB.vertex_shader import *
except:
	print("Error importing GL / shaders")
	sys.exit()

from os.path import isfile, dirname, realpath, join
from PIL import Image
import logging
import numpy as np
import math

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


class Texture( object ):
	"""Texture either loaded from a file."""
	def __init__( self ):
		self.xSize, self.ySize = 0, 0
		self.rawRefence = None

class FileTexture(Texture):
	"""Texture loaded from a file."""
	def __init__( self, fileName ):
		im = Image.open(fileName)
		self.xSize = im.size[0]
		self.ySize = im.size[1]
		self.rawReference=np.array(list(im.getdata()),np.uint8)

class ShaderProgram ( object ):
	"""Manage GLSL programs."""
	def __init__( self ):
		self.__requiredExtensions = ["GL_ARB_fragment_shader",
				 "GL_ARB_vertex_shader",
				 "GL_ARB_shader_objects",
				 "GL_ARB_shading_language_100",
				 "GL_ARB_vertex_shader",
				 "GL_ARB_fragment_shader"]
		self.checkExtensions( self.__requiredExtensions )
		self.__shaderProgramID = glCreateProgramObjectARB()
		self.__checkOpenGLError()
		self.__programReady = False
		self.__isEnabled = False
		self.__shaderObjectList = []

	def checkExtensions( self, extensions ):
		"""Check if all extensions in a list are present."""
		for ext in extensions:
			if ( not ext ):
				print("Driver does not support %s", ext)
				sys.exit()

	def __checkOpenGLError( self ):
		"""Print OpenGL error message."""
		err = glGetError()
		if ( err != GL_NO_ERROR ):
			print('GLERROR: %s', gluErrorString( err ))
			sys.exit()

	def reset( self ):
		"""Disable and remove all shader programs"""
		for shaderID in self.__shaderObjectList:
			glDetachObjectARB( self.__shaderProgramID, shaderID )
			glDeleteObjectARB( shaderID )
			self.__shaderObjectList.remove( shaderID )
			self.__checkOpenGLError( )
		glDeleteObjectARB( self.__shaderProgramID )
		self.__checkOpenGLError( )
		self.__shaderProgramID = glCreateProgramObjectARB()
		self.__checkOpenGLError( )
		self.__programReady = False

	def addShader( self, shaderType, fileName ):
		"""Read a shader program from a file.

		The program is load and compiled"""
		shaderHandle = glCreateShaderObjectARB( shaderType )
		self.__checkOpenGLError( )
		sourceString = open(fileName, 'r').read()
		glShaderSourceARB(shaderHandle, [sourceString] )
		self.__checkOpenGLError( )
		glCompileShaderARB( shaderHandle )
		success = glGetObjectParameterivARB( shaderHandle,
				GL_OBJECT_COMPILE_STATUS_ARB)
		if (not success):
			print(glGetInfoLogARB( shaderHandle ))
			sys.exit( )
		glAttachObjectARB( self.__shaderProgramID, shaderHandle )
		self.__checkOpenGLError( )
		self.__shaderObjectList.append( shaderHandle )

	def linkShaders( self ):
		"""Link compiled shader programs."""
		glLinkProgramARB( self.__shaderProgramID )
		self.__checkOpenGLError( )
		success = glGetObjectParameterivARB( self.__shaderProgramID,
				GL_OBJECT_LINK_STATUS_ARB )
		if (not success):
			print(glGetInfoLogARB(self.__shaderProgramID))
			sys.exit()
		else:
			self.__programReady = True

	def enable( self ):
		"""Activate shader programs."""
		if self.__programReady:
			glUseProgramObjectARB( self.__shaderProgramID )
			self.__isEnabled=True
			self.__checkOpenGLError( )
		else:
			print("Shaders not compiled/linked properly, enable() failed")

	def disable( self ):
		"""De-activate shader programs."""
		glUseProgramObjectARB( 0 )
		self.__isEnabled=False
		self.__checkOpenGLError( )

	def indexOfUniformVariable( self, variableName ):
		"""Find the index of a uniform variable."""
		if not self.__programReady:
			print("\nShaders not compiled/linked properly")
			result = -1
		else:
			result = glGetUniformLocationARB( self.__shaderProgramID, variableName)
			self.__checkOpenGLError( )
		if result < 0:
			print('Variable "%s" not known to the shader', ( variableName ))
			sys.exit( )
		else:
			return result

	def indexOfVertexAttribute( self, attributeName ):
		"""Find the index of an attribute variable."""
		if not self.__programReady:
			print("\nShaders not compiled/linked properly")
			result = -1
		else:
			result = glGetAttribLocationARB( self.__shaderProgramID, attributeName )
			self.__checkOpenGLError( )
		if result < 0:
			print('Attribute "%s" not known to the shader', ( attributeName ))
			sys.exit( )
		else:
			return result

	def isEnabled( self ):
		return self.__isEnabled

	def getShaderProgId(self):
		return self.__shaderProgramID

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
	global window
	# If escape is pressed, kill everything.
	if args[0].decode() == ESCAPE:
		sys.exit()

def display():
	"""Glut display function."""
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glColor3f(1.0, 0.0, 0.0)
	glBegin(GL_QUADS)
#	glTexCoord2f( 0, 1 )
	glVertex3f(-0.5, -0.5, 0)
#	glTexCoord2f( 0, 0 )
	glVertex3f(-0.5, +0.5, 0)
#	glTexCoord2f( 1, 0 )
	glVertex3f(0.5, +0.5, 0)
#	glTexCoord2f( 1, 1 )
	glVertex3f(0.5, -0.5, 0)
	glEnd()
	glutSwapBuffers()

def initTexture(fileName):
	"""Glut init function."""
	try:
		texture = FileTexture(fileName)
	except:
		print('could not open ', fileName, '; using random texture')
		return

	glClearColor(0, 0, 0, 0)
	glShadeModel(GL_SMOOTH)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexImage2D(GL_TEXTURE_2D, 0, 3, texture.xSize, texture.ySize, 0,
							 GL_RGB, GL_UNSIGNED_BYTE, texture.rawReference)
	glEnable(GL_TEXTURE_2D)

def initShaders():
	"""Initialise shaderProg object."""
	global sP
	sP = ShaderProgram( )
	sP.addShader(GL_FRAGMENT_SHADER_ARB, join(dirname(realpath(__file__)),'ipm.frag'))
	glBindFragDataLocation(sP.getShaderProgId(), 0, 'fragColor')

	sP.linkShaders( )

	sP.enable( )

	glUniform1fARB( sP.indexOfUniformVariable("pitch"), 14.0 * math.pi / 180)
	glUniform1fARB( sP.indexOfUniformVariable("yaw"), 0 * math.pi / 180)
	glUniform1fARB( sP.indexOfUniformVariable("fu"), 309.4362)
	glUniform1fARB( sP.indexOfUniformVariable("fv"), 344.2161)
	glUniform1fARB( sP.indexOfUniformVariable("cu"), 317.9034)
	glUniform1fARB( sP.indexOfUniformVariable("cv"), 256.5352)
	glUniform1fARB( sP.indexOfUniformVariable("h"), 2179.8)

	glUniform2fvARB( sP.indexOfUniformVariable("iResolution"), 1, (640, 480))
	glUniformMatrix2fvARB(sP.indexOfUniformVariable("wROI"), 1, GL_FALSE, ((-11048.41, 3933.21), (7204.76,15698.30)))
	glUniformMatrix2fvARB(sP.indexOfUniformVariable("iROI"), 1, GL_FALSE, ((100, 220), (460, 350)))

def main():
	logging.basicConfig(format=FORMAT)
	parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
	args = parse_cmdline(parser)
	log = logging.getLogger("ipm openGL")
	log.setLevel(args.verbose)
	log.info("OpenGL acceleration to compute an IPM")

	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
	glutInitWindowSize(250, 250)
	glutInitWindowPosition(100, 100)
	glutCreateWindow(sys.argv[0])
	glutKeyboardFunc(keyPressed)
	glutDisplayFunc( display )


	initTexture(args.image)
	initShaders()

	glutMainLoop(  )

	# Register the function called when the keyboard is pressed.

print("Hit ESC key to quit.")
main()
