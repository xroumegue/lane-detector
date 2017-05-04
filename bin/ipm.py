# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Author: Nicolas P .Rougier
# Date:   04/03/2014
# -----------------------------------------------------------------------------
import math
import numpy as np

from vispy import app
from vispy.gloo import gl


ipm_vertex = """
#version 450
attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_template = """
#version 450

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iGlobalTime;           // shader playback time (in seconds)
uniform vec4      iMouse;                // mouse pixel coords
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform float     iChannelTime[4];       // channel playback time (in sec)
uniform vec2      iOffset;               // pixel offset for tiled rendering

%s

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy + iOffset);
}
"""



resolution = (1280, 720)

class Canvas(app.Canvas):
    def __init__(self, fragShaderFilename, imageFilename):
        app.Canvas.__init__(self, size=resolution,
                            title='IPM (GL version)',
                            keys='interactive')

        try:
            glsl_shader = open(fragShaderFilename, 'r').read()
        except:
            print("Error while opening fragment shader file")
            sys.exit()

        self.fragmentShader = fragment_template % glsl_shader
        self.vertexShader = ipm_vertex
        im = Image.open(imageFilename)
        self.rawTexture = np.array(list(im.getdata()), np.uint8).reshape(im.size[1], im.size[0], 3)


    def on_initialize(self, event):
        # Build & activate ipm shader program
        self.shader= gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(vertex, self.vertexShader)
        gl.glShaderSource(fragment, self.fragmentShader)
        gl.glCompileShader(vertex)
        gl.glCompileShader(fragment)
        gl.glAttachShader(self.shader, vertex)
        gl.glAttachShader(self.shader, fragment)
        gl.glLinkProgram(self.shader)
        gl.glDetachShader(self.shader, vertex)
        gl.glDetachShader(self.shader, fragment)
        gl.glUseProgram(self.shader)

        # Get data & build cube buffers
        vcube_data, self.icube_data = makecube()

        vertices = np.array[(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]
        vcube = gl.glCreateBuffer()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vcube)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vcube_data, gl.GL_STATIC_DRAW)
        icube = gl.glCreateBuffer()
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, icube)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                        self.icube_data, gl.GL_STATIC_DRAW)

        # Bind cube attributes
        stride = vcube_data.strides[0]
        offset = 0
        loc = gl.glGetAttribLocation(self.cube, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

        offset = vcube_data.dtype["a_position"].itemsize
        loc = gl.glGetAttribLocation(self.cube, "a_texcoord")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

        # Create & bind cube texture
        crate = checkerboard()
        texture = gl.glCreateTexture()
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                           gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                           gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_LUMINANCE, gl.GL_LUMINANCE,
                        gl.GL_UNSIGNED_BYTE, crate.shape[:2])
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, gl.GL_LUMINANCE,
                           gl.GL_UNSIGNED_BYTE, crate)
        loc = gl.glGetUniformLocation(self.cube, "u_texture")
        gl.glUniform1i(loc, texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Create & bind cube matrices
        view = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        projection = np.eye(4, dtype=np.float32)
        translate(view, 0, 0, -7)
        self.phi, self.theta = 60, 20
        rotate(model, self.theta, 0, 0, 1)
        rotate(model, self.phi, 0, 1, 0)
        loc = gl.glGetUniformLocation(self.cube, "u_model")
        gl.glUniformMatrix4fv(loc, 1, False, model)
        loc = gl.glGetUniformLocation(self.cube, "u_view")
        gl.glUniformMatrix4fv(loc, 1, False, view)
        loc = gl.glGetUniformLocation(self.cube, "u_projection")
        gl.glUniformMatrix4fv(loc, 1, False, projection)

        # OpenGL initalization
        gl.glClearColor(0.30, 0.30, 0.35, 1.00)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self._resize(*(self.size + self.physical_size))
        self.timer = app.Timer('auto', self.on_timer, start=True)

    def on_draw(self, event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDrawElements(gl.GL_TRIANGLES, self.icube_data.size,
                          gl.GL_UNSIGNED_INT, None)

    def on_resize(self, event):
        self._resize(*(event.size + event.physical_size))

    def _resize(self, width, height, physical_width, physical_height):
        gl.glViewport(0, 0, physical_width, physical_height)
        projection = perspective(35.0, width / float(height), 2.0, 10.0)
        loc = gl.glGetUniformLocation(self.cube, "u_projection")
        gl.glUniformMatrix4fv(loc, 1, False, projection)

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        model = np.eye(4, dtype=np.float32)
        rotate(model, self.theta, 0, 0, 1)
        rotate(model, self.phi, 0, 1, 0)
        loc = gl.glGetUniformLocation(self.cube, "u_model")
        gl.glUniformMatrix4fv(loc, 1, False, model)
        self.update()

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
