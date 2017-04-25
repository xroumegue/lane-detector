const float M_PI = 3.14159265358;


struct sCameraInfo {
    vec2 focalLength;
    vec2 opticalCenter;
    float height;
    float pitch;
    float yaw;
    float imageWidth;
    float imageHeight;
};

struct sIPMinfo {
    int width;
    int height;
    float left;
    float right;
    float top;
    float bottom;
};

sIPMinfo IPMinfo = sIPMinfo(160, 120, 100.0, 460.0, 220.0, 350.0);
sCameraInfo cameraInfo = sCameraInfo(vec2(309.4362, 344.2161), vec2(317.9034, 256.5352), 2179.8, 14.0 * (M_PI/180.0), 0.0 * (M_PI/180.0), 640.0, 480.0);


vec2 Ti2g(in vec2 _in)
{

float c1 = cos(cameraInfo.pitch);
float s1 = sin(cameraInfo.pitch);
float c2 = cos(cameraInfo.yaw);
float s2 = sin(cameraInfo.yaw);
float fu = cameraInfo.focalLength.x;
float fv = cameraInfo.focalLength.y;
float cu = cameraInfo.opticalCenter.x;
float cv = cameraInfo.opticalCenter.y;
float h = cameraInfo.height;


   vec4 g;
    vec2 g2ret;
    vec4 _in4;
    mat4 Mi2g = transpose(
        			mat4(
                        	-c2/fu,		s1*s2/fv,	(cu*c2/fu) - (cv*s1*s2/fv) - (c1*s2),		0,
                        	s2/fu,		s1*c1/fv,	(-cu*s2/fu) - (cv*s1*c2/fv) - (c1*c2),		0,
                        	0,			c1/fv,		(-cv*c1/fv) +s1,							0,
                        	0,			-c1/(h*fv),	(cv*c1/(h*fv)) - (s1/h),					0
					)
        		);

    _in4.xy = _in.xy;
    _in4.z = 1.0;
    _in4.w = 1.0;


    g = Mi2g * _in4;
	g2ret.xy = g.xy;
    g2ret.xy /= g.w;

    return g2ret;

}


vec2 Tg2i(in vec2 _in)
{

float c1 = cos(cameraInfo.pitch);
float s1 = sin(cameraInfo.pitch);
float c2 = cos(cameraInfo.yaw);
float s2 = sin(cameraInfo.yaw);
float fu = cameraInfo.focalLength.x;
float fv = cameraInfo.focalLength.y;
float cu = cameraInfo.opticalCenter.x;
float cv = cameraInfo.opticalCenter.y;
float h = cameraInfo.height;

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


vec2 getVanishingPoint(sCameraInfo cam)
{
    mat3 transform;

	vec3 vp = vec3(sin(cam.yaw)/cos(cam.pitch), cos(cam.yaw)/cos(cam.pitch), 0);

    mat3 tyaw = transpose(
        			mat3(
        				cos(cam.yaw), -sin(cam.yaw), 0,
        				sin(cam.yaw), cos(cam.yaw), 0,
        				0, 0, 1
					)
    			);

    mat3 tpitch = transpose(
        			mat3(
        				1, 0, 0,
        				0, -sin(cam.pitch), -cos(cam.pitch),
        				0, cos(cam.pitch), -sin(cam.pitch)
                    )
    			);

    mat3 t1 = transpose(
        			mat3(
        				cam.focalLength.x, 0, cam.opticalCenter.x,
        				0, cam.focalLength.y, cam.opticalCenter.y,
        				0, 0, 1
					)
				);

    transform = tpitch * tyaw;
    transform = t1 * transform;
    vp = transform * vp;

    return vec2(vp.x, vp.y);

}



mat2 getROI(in sIPMinfo ipm, sCameraInfo cam)
{
    mat2 roi;

    vec2 vp = getVanishingPoint(cam);

    vec2 center = vec2(vp.x, ipm.top);
    vec2 right = vec2(ipm.right, ipm.top);
    vec2 left = vec2(ipm.left, ipm.top);
    vec2 eye = vec2(vp.x, ipm.bottom);


    vec2 wCenter = Ti2g(center);
    vec2 wRight = Ti2g(right);
    vec2 wLeft = Ti2g(left);
    vec2 wEye = Ti2g(eye);

    roi[0].x = min(wCenter.x,(min(wRight.x, min(wLeft.x, wEye.x))));
    roi[0].y = min(wCenter.y,(min(wRight.y, min(wLeft.y, wEye.y))));

    roi[1].x = max(wCenter.x,(max(wRight.x, max(wLeft.x, wEye.x))));
    roi[1].y = max(wCenter.y,(max(wRight.y, max(wLeft.y, wEye.y))));


    return roi;

}


void main( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
//    uv.y = 1.0 - uv.y;

    vec2 iROI1 = vec2(IPMinfo.left, IPMinfo.top);
    vec2 iROI2 = vec2(IPMinfo.right, IPMinfo.bottom);

    mat2 wROI = getROI(IPMinfo, cameraInfo);

    float xMin = wROI[0].x;
    float xMax = wROI[1].x;
    float yMin = wROI[0].y;
    float yMax = wROI[1].y;


    float yScale = yMax - yMin;
    float xScale = xMax - xMin;

    vec2 w = vec2(xMin + uv.x * xScale, yMin + uv.y * yScale);
	vec2 i = Tg2i(w);

	vec4 outOfROI = vec4(0.0, 0.0, 1.0, 1.0);

    if (i.x < float(IPMinfo.left))
    	fragColor = outOfROI;
    else if (i.x > float(IPMinfo.right))
        fragColor = outOfROI;
	else if (i.y > float(IPMinfo.bottom))
        fragColor = outOfROI;
	else if (i.y < float(IPMinfo.top))
        fragColor = outOfROI;
    else

        i.x /= cameraInfo.imageWidth;
        i.y /= cameraInfo.imageHeight;
		fragColor = texture(iChannel0, i);
}
