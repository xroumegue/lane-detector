#version 450

uniform mat2 wROI;
uniform mat2 iROI;
uniform vec2 iResolution;
uniform sampler2D tex;

uniform float pitch;
uniform float yaw;
uniform float fu;
uniform float fv;
uniform float cu;
uniform float cv;
uniform float h;

in vec4 fragCoord;
out vec4 fragColor;

vec2 Tg2i(in vec2 _in)
{

float c1 = cos(pitch);
float s1 = sin(pitch);
float c2 = cos(yaw);
float s2 = sin(yaw);

    vec2 i2ret;
    vec4 _in4;
    vec4 i;

	_in4.xy = _in.xy;
    _in4.z = -h;
    _in4.w = 1.0;

    mat4 Mg2i = transpose(
        			mat4(
                        	fu*c2 + cu*c1*s2,	cu*c1*c2 - s2*fu,	-cu*s1,				0,
                        	s2*(cv*c1 - fv*s1),	c2*(cv*c1 - fv*s1),	-fv*c1 - cv*s1,		0,
							c1*s2,				c1*c2,				-s1,				0,
							c1*s2,				c1*c2,				-s1,				0
                    )
        		);

    i = Mg2i * _in4;
    i2ret.xy = i.xy / i.z;

    return i2ret;

}

void main()
{
	vec2 uv = fragCoord.xy / iResolution.xy;

    float xMin = wROI[0].x;
    float xMax = wROI[1].x;
    float yMin = wROI[0].y;
    float yMax = wROI[1].y;

    float yScale = yMax - yMin;
    float xScale = xMax - xMin;

    vec2 w = vec2(xMin + uv.x * xScale, yMin + uv.y * yScale);
	vec2 i = Tg2i(w);

	vec4 outOfROI = vec4(0.0, 0.0, 1.0, 1.0);

    if (i.x < iROI[0].x)
    	fragColor = outOfROI;
    else if (i.x > iROI[1].x)
        fragColor = outOfROI;
	else if (i.y > iROI[1].y)
        fragColor = outOfROI;
	else if (i.y < iROI[0].y)
        fragColor = outOfROI;
    else
        i.x /= iResolution.x;
        i.y /= iResolution.y;
		fragColor = texture(tex, i);
}
