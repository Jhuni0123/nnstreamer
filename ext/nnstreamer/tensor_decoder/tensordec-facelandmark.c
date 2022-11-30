#include <glib.h>
#include <gst/gstinfo.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensordecutil.h"

void init_fm (void) __attribute__ ((constructor));
void fini_fm (void) __attribute__ ((destructor));

#define MEDIAPIPE_NUM_FACE_LANDMARKS (468)
#define MEDIAPIPE_NUM_LINES (13)
#define MEDIAPIPE_LINE_WIDTH (1)
#define MEDIAPIPE_POINT_SIZE (2)

/**
 * @brief face landmark model enum
 */
typedef enum {
  MEDIAPIPE_FACE_LANDMARK = 0,

  FACE_LANDMARK_UNKNOWN,
} face_landmark_modes;

/** @brief Internal data structure for face landmark */
typedef struct {
  face_landmark_modes mode; /**< The face landmark decoding mode */

  /* From option2 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option3 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */
} face_landmark_data;

typedef struct {
  float x;
  float y;
  float z;
} landmark_point;

typedef struct {
  int x;
  int y;
} plot_point;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fm_init (void **pdata)
{
  face_landmark_data *data;

  data = *pdata = g_new0 (face_landmark_data, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  data->width = 0;
  data->height = 0;
  data->i_width = 0;
  data->i_height = 0;

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
fm_exit (void **pdata)
{
  face_landmark_data *data = *pdata;
  // TODO: free inner data
  UNUSED (data);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fm_setOption (void **pdata, int opNum, const char *param)
{
  face_landmark_data *data = *pdata;

  if (opNum == 0) {
    /* option1 = face landmark decoding mode */
    // TODO
  } else if (opNum == 1) {
    /* option2 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR ("mode-option-2 of facelandmark is output video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE; /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING ("mode-option-2 of facelandmark is output video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->width = dim[0];
    data->height = dim[1];
    return TRUE;
  } else if (opNum == 2) {
    /* option3 = input video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR ("mode-option-3 of facelandmark is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE; /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING ("mode-option-3 of facelandmark is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->i_width = dim[0];
    data->i_height = dim[1];
    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/**
 * @brief check the num_tensors is valid
 */
static int
_check_tensors (const GstTensorsConfig *config, const unsigned int limit)
{
  unsigned int i;
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (config->info.num_tensors >= limit, FALSE);
  if (config->info.num_tensors > limit) {
    GST_WARNING ("tensor-decoder:boundingbox accepts %d or less tensors. "
                 "You are wasting the bandwidth by supplying %d tensors.",
        limit, config->info.num_tensors);
  }

  /* tensor-type of the tensors shoule be the same */
  for (i = 1; i < config->info.num_tensors; ++i) {
    g_return_val_if_fail (config->info.info[i - 1].type == config->info.info[i].type, FALSE);
  }
  return TRUE;
}

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 */
static GstCaps *
fm_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  face_landmark_data *data = *pdata;
  GstCaps *caps;
  int i;
  char *str;

  if (data->mode == MEDIAPIPE_FACE_LANDMARK) {
    const guint *dim1 = config->info.info[0].dimension;
    const guint *dim2 = config->info.info[1].dimension;
    if (!_check_tensors (config, 2U))
      return NULL;

    g_return_val_if_fail (dim1[0] == 1404, NULL);
    for (i = 1; i < 4; ++i)
      g_return_val_if_fail (dim1[i] == 1, NULL);
    for (i = 0; i < 4; ++i)
      g_return_val_if_fail (dim2[i] == 1, NULL);
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, width = %u, height = %u",
      data->width, data->height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);

  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
fm_getTransformSize (void **pdata, const GstTensorsConfig *config,
    GstCaps *caps, size_t size, GstCaps *othercaps, GstPadDirection direction)
{
  UNUSED (pdata);
  UNUSED (config);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  UNUSED (direction);

  return 0;
}

static void
draw_point (uint32_t *frame, face_landmark_data *fmdata, int px, int py, int r, uint32_t color)
{
  int i, j, x, y;
  for (i = -r; i <= r; i++) {
    for (j = -r; j <= r; j++) {
      x = px + i;
      y = py + j;
      if (x < 0 || x > (int) fmdata->width || y < 0 || y > (int) fmdata->height)
        continue;
      frame[y * fmdata->width + x] = color;
    }
  }
}

// Bresenham's line algorithm
static void
draw_line (uint32_t *frame, face_landmark_data *fmdata, int x0, int y0, int x1, int y1)
{
  int dx, sx, dy, sy, error;
  dx = ABS (x1 - x0);
  sx = x0 < x1 ? 1 : -1;
  dy = -ABS (y1 - y0);
  sy = y0 < y1 ? 1 : -1;
  error = dx + dy;

  while (TRUE) {
    draw_point (frame, fmdata, x0, y0, MEDIAPIPE_LINE_WIDTH, 0xFFFF0000);
    if (x0 == x1 && y0 == y1)
      break;
    if (error * 2 >= dy) {
      if (x0 == x1)
        break;
      error += dy;
      x0 += sx;
    }
    if (error * 2 <= dx) {
      if (y0 == y1)
        break;
      error += dx;
      y0 += sy;
    }
  }
}

static void
draw_lines (uint32_t *frame, face_landmark_data *fmdata, GArray *points,
    const uint32_t *point_idx, int point_idx_len)
{
  plot_point *p0, *p1;
  int i;

  // printf("num points = %d\n", point_idx_len);
  for (i = 0; i < point_idx_len - 1; i++) {
    // printf("%d %d\n", point_idx[i], point_idx[i+1]);
    p0 = &g_array_index (points, plot_point, point_idx[i]);
    p1 = &g_array_index (points, plot_point, point_idx[i + 1]);
    // printf("%d %d %d %d\n", p0->x, p0->y, p1->x, p1->y);
    draw_line (frame, fmdata, p0->x, p0->y, p1->x, p1->y);
  }
}

/**
 * @brief Draw with the given results (landmark_points[MEDIAPIPE_NUM_FACE_LANDMARKS]) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] fmdata The face-landmark internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo *out_info, face_landmark_data *fmdata, GArray *results)
{
  const uint32_t silhouette[] = { 10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10 };
  const uint32_t lipsUpperOuter[] = { 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291 };
  const uint32_t lipsLowerOuter[] = { 146, 91, 181, 84, 17, 314, 405, 321, 375, 291 };
  const uint32_t lipsUpperInner[] = { 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308 };
  const uint32_t lipsLowerInner[] = { 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308 };

  const uint32_t rightEyeUpper0[] = { 246, 161, 160, 159, 158, 157, 173 };
  const uint32_t rightEyeLower0[] = { 33, 7, 163, 144, 145, 153, 154, 155, 133 };

  const uint32_t rightEyebrowUpper[] = { 70, 63, 105, 66, 107 };
  const uint32_t rightEyebrowLower[] = { 46, 53, 52, 65, 55 };

  const uint32_t leftEyeUpper0[] = { 466, 388, 387, 386, 385, 384, 398 };
  const uint32_t leftEyeLower0[] = { 263, 249, 390, 373, 374, 380, 381, 382, 362 };

  const uint32_t leftEyebrowUpper[] = { 300, 293, 334, 296, 336 };
  const uint32_t leftEyebrowLower[] = { 276, 283, 282, 295, 285 };

  const uint32_t *lines[MEDIAPIPE_NUM_LINES] = {
    silhouette,
    lipsUpperOuter,
    lipsLowerOuter,
    lipsUpperInner,
    lipsLowerInner,
    rightEyeUpper0,
    rightEyeLower0,
    rightEyebrowUpper,
    rightEyebrowLower,
    leftEyeUpper0,
    leftEyeLower0,
    leftEyebrowUpper,
    leftEyebrowLower,
  };
  const int lines_len[MEDIAPIPE_NUM_LINES]
      = { 37, 11, 10, 11, 11, 7, 9, 5, 5, 7, 9, 5, 5 };

  GArray *points = g_array_sized_new (
      FALSE, TRUE, sizeof (plot_point), MEDIAPIPE_NUM_FACE_LANDMARKS);
  int i;
  plot_point *pp;

  uint32_t *frame = (uint32_t *) out_info->data;
  unsigned int arr_i;

  for (arr_i = 0; arr_i < results->len; arr_i++) {
    int x, y;
    landmark_point *p = &g_array_index (results, landmark_point, arr_i);

    x = (int) ((fmdata->width * p->x) / fmdata->i_width);
    y = (int) ((fmdata->height * p->y) / fmdata->i_height);
    x = MAX (0, x);
    y = MAX (0, y);
    x = MIN ((int) fmdata->width - 1, x);
    y = MIN ((int) fmdata->height - 1, y);
    g_array_append_val (points, ((plot_point){ .x = x, .y = y }));
  }

  // Draw lines
  for (i = 0; i < MEDIAPIPE_NUM_LINES; i++) {
    draw_lines (frame, fmdata, points, lines[i], lines_len[i]);
  }

  // Draw points
  for (i = 0; i < MEDIAPIPE_NUM_FACE_LANDMARKS; i++) {
    pp = &g_array_index (points, plot_point, i);
    draw_point (frame, fmdata, pp->x, pp->y, MEDIAPIPE_POINT_SIZE, 0xFF0000FF);
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
fm_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  face_landmark_data *data = *pdata;
  const size_t size = (size_t) data->width * data->height * 4; /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *results = NULL;
  const guint num_tensors = config->info.num_tensors;
  gboolean need_output_alloc;

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  /* Ensure we have outbuf properly allocated */
  if (need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-face_landmark.\n");
    goto error_free;
  }

  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  {
    const GstTensorMemory *landmarks;
    float *landmarks_input;
    size_t i;
    // float * faceflag = (float *) (&input[1])->data;

    g_assert (num_tensors == 2);
    results = g_array_sized_new (
        FALSE, TRUE, sizeof (landmark_point), MEDIAPIPE_NUM_FACE_LANDMARKS);

    landmarks = &input[0];
    landmarks_input = (float *) landmarks->data;

    for (i = 0; i < MEDIAPIPE_NUM_FACE_LANDMARKS; i++) {
      float x = landmarks_input[i * 3];
      float y = landmarks_input[i * 3 + 1];
      float z = landmarks_input[i * 3 + 2];
      landmark_point point = { .x = x, .y = y, .z = z };
      g_array_append_val (results, point);
    }
  }

  draw (&out_info, data, results);
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);
  if (need_output_alloc) {
    gst_buffer_append_memory (outbuf, out_mem);
  } else {
    gst_memory_unref (out_mem);
  }

  return GST_FLOW_OK;

error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

static gchar decoder_subplugin_face_landmark[] = "face_landmark";

/** @brief Face Landmark tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef faceLandmark = {
  .modename = decoder_subplugin_face_landmark,
  .init = fm_init,
  .exit = fm_exit,
  .setOption = fm_setOption,
  .getOutCaps = fm_getOutCaps,
  .getTransformSize = fm_getTransformSize,
  .decode = fm_decode,
};

/** @brief Initialize this object for tensordec-plugin */
void
init_fm (void)
{
  nnstreamer_decoder_probe (&faceLandmark);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_fm (void)
{
  nnstreamer_decoder_exit (faceLandmark.modename);
}
