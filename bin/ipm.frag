#version 450

uniform vec3      	iResolution;
uniform sampler2D 	iChannel;
uniform vec3      	iChannelResolution;
uniform mat4		uMg2i;
uniform mat4		uMi2g;
uniform mat2		uwROI;
uniform mat2		uIPM;
uniform float		uH;

vec2 Ti2g(in vec2 _in)
{
        vec4 g;
        vec2 g2ret;
        vec4 _in4;
        _in4.xy = _in.xy;
        _in4.z = 1.0;
        _in4.w = 1.0;

        g = transpose(uMi2g) * _in4;
        g2ret.xy = g.xy;
        g2ret.xy /= g.w;

        return g2ret;
}

vec2 Tg2i(in vec2 _in)
{
        vec2 i2ret;
        vec4 _in4;
        vec4 i;

        _in4.xy = _in.xy;
        _in4.z = -uH;
        _in4.w = 1.0;

        i = transpose(uMg2i) * _in4;
        i2ret.xy = i.xy / i.z;

        return i2ret;
}

void main()
{
        vec2 uv = gl_FragCoord.xy / iResolution.xy;
        float xMin = uwROI[0].x;
        float xMax = uwROI[1].x;
        float yMin = uwROI[0].y;
        float yMax = uwROI[1].y;

        float yScale = yMax - yMin;
        float xScale = xMax - xMin;

        vec2 w = vec2(xMin + uv.x * xScale, yMin + uv.y * yScale);
        vec2 i = Tg2i(w);

        vec4 outOfROI = vec4(0.0, 0.0, 0.0, 1.0);

        if (i.x < uIPM[0].x)
                gl_FragColor = outOfROI;
        else if (i.x > uIPM[1].x)
                gl_FragColor = outOfROI;
        else if (i.y > uIPM[0].y)
                gl_FragColor = outOfROI;
        else if (i.y < uIPM[1].y)
                gl_FragColor = outOfROI;
        else {
                i.xy /= iChannelResolution.xy;
                gl_FragColor = texture(iChannel, i);
        }
}
