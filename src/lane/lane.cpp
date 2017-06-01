#include "opencv_camera_display.h"

#include <VX/vx.h>
#include <getopt.h>
/*
 * Useful macros for OpenVX error checking:
 *   ERROR_CHECK_STATUS     - check whether the status is VX_SUCCESS
 *   ERROR_CHECK_OBJECT     - check whether the object creation is successful
 */
#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
		printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
		exit(1); \
	} \
}

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
		printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
		exit(1); \
	} \
}

/*
 * log_callback() function implements a mechanism to print log messages
 * from OpenVX framework onto console.
 */
void VX_CALLBACK log_callback( vx_context    context,
                               vx_reference  ref,
                               vx_status     status,
                               const vx_char string[] )
{
	printf( "LOG: [ status = %d ] %s\n", status, string );
	fflush( stdout );
}

// width and height for intermidiate images
#define INTERNAL_WIDTH  220
#define INTERNAL_HEIGHT 240

// Below are several parameters to define road area for mapping and processing.

// Relative height of vanishing point or distance from bottom camera view border to horizon line.
// It is supposed that horizon is parallel to camera view borders
#define REMAP_HORIZON      0.6458f

// Relative height of point where left and right lines can be distinguished.
// It is distance from bottom camera view border
// to the farest point on the road that will be mapped.
// It has to be slightly less than REMAP_HORIZON
#define REMAP_FAR_CLIP      ((REMAP_HORIZON)-0.10f)

// Relative height of hood’s front not to detect line segments on it.
// It is distance from bottom camera border
// to the nearest point on the road that will be mapped.
#define REMAP_NEAR_CLIP     0.2708f

// Relative width of the mapped road area for the REMAP_NEAR_CLIP
// This value correlates with road width that will be processed
#define REMAP_NEAR_WIDTH    0.95f
//   ^
//   |               Camera View
//   +1.0 +------------------------------------+------- top border (1.0)
//   |    |                                    |
//   |    |                                    |
//   |    |                                    |
//   |    |                                    |
//   |    |-----------------.------------------|-'-REMAP_HORIZON (0.435)
//   +0.5 |                . .                 | |
//   |    |               1---3 - - - - - - - -|-+-'-REMAP_FAR_CLIP (0.385)
//   |    |              /M A P\               | | |
//   |    |             /A R E A\              | | |
//   |    |            0---------2 - - - - - - |-+-+-'-REMAP_NEAR_CLIP (0.17)
//   |    |                                    | | | |
//   +0.0 +------------------------------------+-'-'-'-- bottom border (0.0)
//                     |         |
//                     '---------'
//                     NEAR_WIDTH
//
//

/*
 *  Canny edge detector
 */

#define CANNY_SOBEL_FILTER_SIZE 3
#define CANNY_THRESH_MIN 160
#define CANNY_THRESH_MAX 180

static void calcLaneArea(int width, int height, cv::Point2f laneArea[4])
{
    // Calc lane area from REMAP_HORIZON, REMAP_FAR_CLIP, REMAP_NEAR_CLIP and REMAP_NEAR_WIDTH parameters.
    // In general any 4 points can be defined below
    float dxFar = 0.5f * REMAP_NEAR_WIDTH * (REMAP_FAR_CLIP - REMAP_HORIZON) / (REMAP_NEAR_CLIP - REMAP_HORIZON);
    float dxNear = 0.5f * REMAP_NEAR_WIDTH;
    laneArea[0].x = (0.5f-dxNear)*width; laneArea[0].y = (1.0f-REMAP_NEAR_CLIP)*height;
    laneArea[1].x = (0.5f-dxFar) *width; laneArea[1].y = (1.0f-REMAP_FAR_CLIP) *height;
    laneArea[2].x = (0.5f+dxNear)*width; laneArea[2].y = (1.0f-REMAP_NEAR_CLIP)*height;
    laneArea[3].x = (0.5f+dxFar) *width; laneArea[3].y = (1.0f-REMAP_FAR_CLIP) *height;
}
// function that calculate perspective transformation matrix from camera (width,height) map area to
// whole internal top view (0,0)-(INTERNAL_WIDTH, INTERNAL_HEIGHT)
static cv::Mat calcPerspectiveTransform(int width, int height, int outWidth, int outHeight)
{// calc perspective transform from 4 points
    // define persepctive transform by 4 points placed on road
    cv::Point2f srcP[4];
    calcLaneArea(width,height,srcP);
    cv::Point2f dstP[4] =
    {
        {0                , (float)(outHeight)},
        {0                , 0},
        {(float)(outWidth), (float)(outHeight)},
        {(float)(outWidth), 0},
    };
    return cv::getPerspectiveTransform(dstP,srcP);
}

static int vxShowImage(vx_image _img, const char *name)
{
		vx_uint32  _width, _height;
		ERROR_CHECK_STATUS(vxQueryImage(_img, VX_IMAGE_WIDTH, &_width, sizeof(vx_uint32)));
		ERROR_CHECK_STATUS(vxQueryImage(_img, VX_IMAGE_HEIGHT, &_height, sizeof(vx_uint32)));
		vx_rectangle_t rect = { 0, 0, _width, _height };
		vx_map_id map_id;
		vx_imagepatch_addressing_t addr;
		void * ptr;
		ERROR_CHECK_STATUS(
			vxMapImagePatch(
				_img,
				&rect,
				0,
				&map_id,
				&addr,
				&ptr,
				VX_READ_ONLY,
				VX_MEMORY_TYPE_HOST,
				VX_NOGAP_X
			)
		);
		cv::Mat mat(_height, _width, CV_8U, ptr, addr.stride_y);
		cv::imshow( name, mat );
		ERROR_CHECK_STATUS(
			vxUnmapImagePatch(
				_img,
				map_id)
		);
}

/*
 * Command-line usage:
 *   % lane [<video-sequence>|<camera-device-number>]
 */
int main( int argc, char * argv[] )
{
	static struct {
		const char *video_sequence;
		int verbose_flag;
		int help_flag;
	} opts = {
		.video_sequence = NULL,
		.verbose_flag = 0,
	};

	while(1) {
		int option_index, c;
		static struct option long_options[] = {
			/* These options set a flag. */
			{"verbose", no_argument, &opts.verbose_flag, 1},
			{"help", no_argument, &opts.help_flag, 1},
			/* These options don't set a flag. */
			{"file", required_argument, 0, 'f'},
			{0, 0, 0, 0}
		};
		c = getopt_long(argc, argv, "vhf:", long_options, &option_index);
		if (c == -1)
			break;
		switch(c) {
			case 0:
				printf("%s mode activated\n", long_options[option_index].name);
			break;
			case 'f':
				opts.video_sequence = optarg;
				printf("video file: %s\n", optarg);
			break;
			default:
				printf("Going to abort: %c", c);
				abort();
		}
	}

	CGuiModule gui( opts.video_sequence );

/*
	Try to grab the first video frame from the sequence using cv::VideoCapture
	and check if a video frame is available.
*/
	if(!gui.Grab()) {
	        printf( "ERROR: input has no video\n" );
	        return 1;
	}

	/* Grab some constants */
	vx_uint32 width = gui.GetWidth();
	vx_uint32 height = gui.GetHeight();
	vx_uint32 stride = gui.GetStride();
	int internalWidth = width;
	int internalHeight= height;

	/* Create vx context */
	vx_context context = vxCreateContext();
	ERROR_CHECK_OBJECT(context);
	/* Register log callback */
	vxRegisterLogCallback( context, log_callback, vx_false_e );
	vxAddLogEntry( ( vx_reference ) context, VX_FAILURE, "OpenVX something application\n" );

	/* Create a graph */
	vx_graph graph = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph);

	/* Create openvx image objects */
	/*		.... RGB input image */
	vx_image input_rgb_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
	ERROR_CHECK_OBJECT(input_rgb_image);
	/*		.... GRAYSCALE image */
	vx_image luma_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
	ERROR_CHECK_OBJECT(luma_image);
	/*		... WARP (IPM) image */
	vx_image ipm_image = vxCreateImage(context, internalWidth, internalHeight, VX_DF_IMAGE_U8);
	ERROR_CHECK_OBJECT(ipm_image);

	/*		... canny image */
	vx_image canny_image = vxCreateImage(context, internalWidth, internalHeight, VX_DF_IMAGE_U8);
	ERROR_CHECK_OBJECT(canny_image);

	/*		.... YUV (virtual) image */
	vx_image yuv_image  = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_IYUV );
	ERROR_CHECK_OBJECT(yuv_image);


        cv::Mat ocvH;
        calcPerspectiveTransform(width, height, internalWidth, internalHeight).convertTo(ocvH, CV_32F);
        vx_matrix ovxH = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
        vx_float32  data[9] =
        {
            ocvH.at<float>(0,0),ocvH.at<float>(1,0),ocvH.at<float>(2,0),
            ocvH.at<float>(0,1),ocvH.at<float>(1,1),ocvH.at<float>(2,1),
            ocvH.at<float>(0,2),ocvH.at<float>(1,2),ocvH.at<float>(2,2)
        };
        printf("Warp Perspective Matrix = 9::%f,%f,%f,%f,%f,%f,%f,%f,%f\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8]);
        ERROR_CHECK_STATUS( vxCopyMatrix(ovxH, &data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );

	/*  Canny edge detector */

	vx_int32 ovxCannyGradientSize = CANNY_SOBEL_FILTER_SIZE;
	vx_uint32 ovxThreshCannyMin = CANNY_THRESH_MIN;
	vx_uint32 ovxThreshCannyMax = CANNY_THRESH_MAX;
	vx_threshold ovxThreshCanny = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
	vxSetThresholdAttribute(ovxThreshCanny, VX_THRESHOLD_THRESHOLD_LOWER, &ovxThreshCannyMin,
	                        sizeof(ovxThreshCannyMin));
	vxSetThresholdAttribute(ovxThreshCanny, VX_THRESHOLD_THRESHOLD_UPPER, &ovxThreshCannyMax,
                            sizeof(ovxThreshCannyMax));

	/* Create nodes */
	vx_node nodes[] = {
		vxColorConvertNode(graph, input_rgb_image, yuv_image),
		vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
		vxWarpPerspectiveNode(graph, luma_image, ovxH, VX_INTERPOLATION_BILINEAR, ipm_image),
		vxCannyEdgeDetectorNode(graph, ipm_image, ovxThreshCanny, ovxCannyGradientSize, VX_NORM_L1, canny_image),
	};

	/* Check each node.. and release it since already refcounted by graph */
	for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ ) {
		ERROR_CHECK_OBJECT( nodes[i] );
		ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
	}
	/* Release (internal) images since already refcounted by nodes */
	ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ) );

	/* Verify the graph */
	ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );

	for(int frame_index = 0; !gui.AbortRequested(); frame_index++) {
		/* openCV to openVX image copy parameters */
		vx_rectangle_t cv_rgb_image_region = {
			.start_x    = 0,
			.start_y    = 0,
			.end_x      = width,
			.end_y      = height,
		};

		vx_imagepatch_addressing_t cv_rgb_image_layout;
		cv_rgb_image_layout.stride_x   = 3;
		cv_rgb_image_layout.stride_y   = stride;

		/* Get the image from OpenCV */
		vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();

		/* Copy image to openVX buffer */
		ERROR_CHECK_STATUS(
			vxCopyImagePatch(
				input_rgb_image,
				&cv_rgb_image_region,
				0,
				&cv_rgb_image_layout,
				cv_rgb_image_buffer,
				VX_WRITE_ONLY,
				VX_MEMORY_TYPE_HOST
			)
		);
		/* Process the graph */
		ERROR_CHECK_STATUS(
			vxProcessGraph(graph)
		);
		/* Get output image */
		vxShowImage(luma_image, "Grayscale");
		vxShowImage(ipm_image, "IPM");
		vxShowImage(canny_image, "Canny");
		gui.Show();
		if(!gui.Grab()) {
			/* Terminate the processing loop if the end of sequence is detected. */
			gui.WaitForKey();
			break;
		}
	}

	/* Clean the dirty place */
	ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
	ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
	ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
	ERROR_CHECK_STATUS(vxReleaseImage(&ipm_image));
	ERROR_CHECK_STATUS(vxReleaseImage(&canny_image));
	ERROR_CHECK_STATUS(vxReleaseContext(&context));

	return 0;
}
