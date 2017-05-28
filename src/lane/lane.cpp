#include "opencv_camera_display.h"

#include <VX/vx.h>
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
/*
 * Command-line usage:
 *   % lane [<video-sequence>|<camera-device-number>]
 */
int main( int argc, char * argv[] )
{
	const char * video_sequence = argv[1];
	CGuiModule gui( video_sequence );

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


	/*		.... YUV (virtual) image */
	vx_image yuv_image  = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_IYUV );
	ERROR_CHECK_OBJECT(yuv_image);

	/* Create nodes */
	vx_node nodes[] = {
	    vxColorConvertNode(graph, input_rgb_image, yuv_image),
	    vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
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
		vx_rectangle_t rect = { 0, 0, width, height };
		vx_map_id map_id;
		vx_imagepatch_addressing_t addr;
		void * ptr;
		ERROR_CHECK_STATUS(
			vxMapImagePatch(
				luma_image,
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
		cv::Mat mat(height, width, CV_8U, ptr, addr.stride_y);
		cv::imshow( "Grayscale", mat );
		ERROR_CHECK_STATUS(
			vxUnmapImagePatch(
				luma_image,
				map_id)
		);

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
	ERROR_CHECK_STATUS(vxReleaseContext(&context));

	return 0;
}
