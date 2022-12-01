/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "face landmark"
 * Copyright (C) 2022 Jonghun Park <whdgnsdl887@gmail.com>
 * Copyright (C) 2022 Youngchan Lee <youngchan1115@gmail.com>
 * Copyright (C) 2022 Changmin Choi <cmchoi9901@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	      tensordec-facelandmark.c
 * @date	      30 Nov 2022
 * @brief	      NNStreamer tensor-decoder subplugin, "face landmark",
 *              which converts tensors to video stream w/ face landmarks
 *              on transparent background.
 *
 * @see		      https://github.com/nnstreamer/nnstreamer
 * @author      Jonghun Park <whdgnsdl887@gmail.com>
 *              Youngchan Lee <youngchan1115@gmail.com>
 *              Changmin Choi <cmchoi9901@gmail.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Decoder mode of face landmark.
 *          Available: mediapipe-face-mesh
 * option2: Decoder mode dependent options
 *          [mediapipe-face-mesh]: face probability threshold
 * option3: Output video size (width, height)
 * option4: Input video size (width, height)
 *
 */

#include <glib.h>
#include <gst/gstinfo.h>
#include <math.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensordecutil.h"

void init_fl (void) __attribute__ ((constructor));
void fini_fl (void) __attribute__ ((destructor));

#define MEDIAPIPE_NUM_FACE_LANDMARKS (468)
#define MEDIAPIPE_NUM_LINES (13)
#define MEDIAPIPE_LINE_WIDTH (1)
#define MEDIAPIPE_POINT_SIZE (2)
#define _sigmoid(x) (1.f / (1.f + expf (-x)))

/**
 * @brief face landmark model enum
 */
typedef enum {
  MEDIAPIPE_FACE_MESH = 0,

  FACE_LANDMARK_UNKNOWN,
} face_landmark_modes;

/**
 * @brief List of face-landmark decoding schemes in string
 */
static const char *fl_modes[] = {
  [MEDIAPIPE_FACE_MESH] = "mediapipe-face-mesh",
  NULL,
};

/** @brief Represents a landmark point */
typedef struct {
  int x;
  int y;
  gfloat z;   /* Optional z-axis coordinate */
} landmark_point;

typedef struct {
  /* face probability */
  float prob;

  /* landmark points */
  GArray *points; /** array of landmark point */

  /* lines */
  const uint32_t **lines; /**< idx 2d array of lines. length is NUM_LANDMARK */
  const int *lines_len; /**< len of each array in lines */
} face_info;

/** @brief Internal data structure for face landmark */
typedef struct {
  /* From option1 */
  face_landmark_modes mode; /**< The face landmark decoding mode */

  /* From option2 */
  float prob_threshold;

  /* visualizing */
  guint line_width;
  guint point_size;

  /* From option3 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option4 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */
} face_landmark_data;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fl_init (void **pdata)
{
  face_landmark_data *fldata;

  fldata = *pdata = g_new0 (face_landmark_data, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  fldata->mode = FACE_LANDMARK_UNKNOWN;
  fldata->prob_threshold = 0.5;
  fldata->width = 0;
  fldata->height = 0;
  fldata->i_width = 0;
  fldata->i_height = 0;

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
fl_exit (void **pdata)
{
  face_landmark_data *fldata = *pdata;
  // TODO: free inner data
  UNUSED (fldata);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fl_setOption (void **pdata, int opNum, const char *param)
{
  face_landmark_data *fldata = *pdata;

  if (opNum == 0) {
    /* option1 = face landmark decoding mode */
    fldata->mode = find_key_strv (fl_modes, param);
    if (fldata->mode == MEDIAPIPE_FACE_MESH) {
      fldata->line_width = MEDIAPIPE_LINE_WIDTH;
      fldata->point_size = MEDIAPIPE_POINT_SIZE;
    }
  } else if (opNum == 1) {
    if (fldata->mode == MEDIAPIPE_FACE_MESH) {
      fldata->prob_threshold = strtod (param, NULL);
    }
  } else if (opNum == 2) {
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
    fldata->width = dim[0];
    fldata->height = dim[1];
    return TRUE;
  } else if (opNum == 3) {
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
    fldata->i_width = dim[0];
    fldata->i_height = dim[1];
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
 *
 * [mediapipe-face-mesh]
 * The first tensor is points of face landmarks.
 *    (3 * MEDIAPIPE_NUM_FACE_LANDMARKS) : 1 : 1 : 1
 * The second tensor is likelihood of face being present
 *    changed to probability in fl_decode
 *    1 : 1 : 1 : 1
 */
static GstCaps *
fl_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  face_landmark_data *fldata = *pdata;
  GstCaps *caps;
  int i;
  char *str;

  if (fldata->mode == MEDIAPIPE_FACE_MESH) {
    const guint *dim1 = config->info.info[0].dimension;
    const guint *dim2 = config->info.info[1].dimension;
    if (!_check_tensors (config, 2U))
      return NULL;

    g_return_val_if_fail (dim1[0] == 3 * MEDIAPIPE_NUM_FACE_LANDMARKS, NULL);
    for (i = 1; i < 4; ++i)
      g_return_val_if_fail (dim1[i] == 1, NULL);
    for (i = 0; i < 4; ++i)
      g_return_val_if_fail (dim2[i] == 1, NULL);
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, width = %u, height = %u",
      fldata->width, fldata->height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);

  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
fl_getTransformSize (void **pdata, const GstTensorsConfig *config,
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

/**
 * @brief draw one point in square
 * @param[out] frame The frame to be drawn
 * @param[in] fldata The face-landmark internal data
 * @param[in] px x-coordinate of point
 * @param[in] py y-coordinate of point
 * @param[in] r size of square (2 * r) by (2 * r)
 * @param[in] color color of point (RGBA)
 */
static void
draw_point (uint32_t *frame, face_landmark_data *fldata, int px, int py, int r, uint32_t color)
{
  int i, j, x, y;
  for (i = -r; i <= r; i++) {
    for (j = -r; j <= r; j++) {
      x = px + i;
      y = py + j;
      if (x < 0 || x > (int) fldata->width || y < 0 || y > (int) fldata->height)
        continue;
      frame[y * fldata->width + x] = color;
    }
  }
}

/**
 * @brief draw one line between two points using Bresenham's line algorithm
 * @param[out] frame The frame to be drawn
 * @param[in] fldata The face-landmark internal data
 * @param[in] x0 x-coordinate of point-0
 * @param[in] y0 y-coordinate of point-0
 * @param[in] x1 x-coordinate of point-1
 * @param[in] y1 y-coordinate of point-1
 */
static void
draw_line (uint32_t *frame, face_landmark_data *fldata, int x0, int y0, int x1, int y1)
{
  int dx, sx, dy, sy, error;
  dx = ABS (x1 - x0);
  sx = x0 < x1 ? 1 : -1;
  dy = -ABS (y1 - y0);
  sy = y0 < y1 ? 1 : -1;
  error = dx + dy;

  while (TRUE) {
    draw_point (frame, fldata, x0, y0, fldata->line_width, 0xFFFF0000);
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

/**
 * @brief draw one line between two points using Bresenham's line algorithm
 * @param[out] frame The frame to be drawn
 * @param[in] fldata The face-landmark internal data
 * @param[in] face The face information includes landmarks
 * @param[in] point_idx The array of idx in points to be drawn
 * @param[in] point_idx_len The length of point_idx
 */
static void
draw_lines (uint32_t *frame, face_landmark_data *fldata, face_info *face,
    const uint32_t *point_idx, int point_idx_len)
{
  landmark_point *p0, *p1;
  int i;

  for (i = 0; i < point_idx_len - 1; i++) {
    p0 = &g_array_index (face->points, landmark_point, point_idx[i]);
    p1 = &g_array_index (face->points, landmark_point, point_idx[i + 1]);
    draw_line (frame, fldata, p0->x, p0->y, p1->x, p1->y);
  }
}

/**
 * @brief Draw with the given face info to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] fldata The face-landmark internal data.
 * @param[in] face The face information includes landmarks
 */
static void
draw (GstMapInfo *out_info, face_landmark_data *fldata, face_info *face)
{
  int i;
  landmark_point *pp;

  uint32_t *frame = (uint32_t *) out_info->data;

  if (face->prob > fldata->prob_threshold) {
    // Draw lines
    for (i = 0; i < MEDIAPIPE_NUM_LINES; i++) {
      draw_lines (frame, fldata, face, face->lines[i], face->lines_len[i]);
    }

    // Draw points
    for (i = 0; i < MEDIAPIPE_NUM_FACE_LANDMARKS; i++) {
      pp = &g_array_index (face->points, landmark_point, i);
      draw_point (frame, fldata, pp->x, pp->y, fldata->point_size, 0xFF0000FF);
    }
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
fl_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  face_landmark_data *fldata = *pdata;
  face_info face;
  const size_t size = (size_t) fldata->width * fldata->height * 4; /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *points = NULL;
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
  face.prob = 1.0;

  if (fldata->mode == MEDIAPIPE_FACE_MESH) {
    const GstTensorMemory *prob;
    const GstTensorMemory *landmarks;
    float *landmarks_input;
    size_t i;

    /** idx of lines which connecting landmarks */
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

    g_assert (num_tensors == 2);

    /* handling landmark points */
    points = g_array_sized_new (FALSE, TRUE, sizeof (landmark_point), MEDIAPIPE_NUM_FACE_LANDMARKS);

    landmarks = &input[0];
    prob = &input[1];
    landmarks_input = (float *) landmarks->data;
    face.prob = _sigmoid(((float *) prob->data)[0]);
    for (i = 0; i < MEDIAPIPE_NUM_FACE_LANDMARKS; i++) {
      landmark_point p = { .x = 0, .y = 0, .z = 0.0f };
      int x, y;
      float lx = landmarks_input[i * 3];
      float ly = landmarks_input[i * 3 + 1];
      /** z-axis depth is not used */
      x = (int) ((fldata->width * lx) / fldata->i_width);
      y = (int) ((fldata->height * ly) / fldata->i_height);
      x = MAX (0, x);
      y = MAX (0, y);
      p.x = MIN ((int) fldata->width - 1, x);
      p.y = MIN ((int) fldata->height - 1, y);
      g_array_append_val (points, p);
    }

    face.points = points;
    face.lines = lines;
    face.lines_len = lines_len;
  } else {
    GST_ERROR ("Failed to get output buffer, unknown mode %d.", fldata->mode);
    goto error_unmap;
  }

  draw (&out_info, fldata, &face);
  g_array_free (points, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  if (need_output_alloc) {
    gst_buffer_append_memory (outbuf, out_mem);
  } else {
    gst_memory_unref (out_mem);
  }

  return GST_FLOW_OK;

error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

static gchar decoder_subplugin_face_landmark[] = "face_landmark";

/** @brief Face Landmark tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef faceLandmark = {
  .modename = decoder_subplugin_face_landmark,
  .init = fl_init,
  .exit = fl_exit,
  .setOption = fl_setOption,
  .getOutCaps = fl_getOutCaps,
  .getTransformSize = fl_getTransformSize,
  .decode = fl_decode,
};

/** @brief Initialize this object for tensordec-plugin */
void
init_fl (void)
{
  nnstreamer_decoder_probe (&faceLandmark);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_fl (void)
{
  nnstreamer_decoder_exit (faceLandmark.modename);
}
