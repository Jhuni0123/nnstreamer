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

/** @brief Internal data structure for face mesh */
typedef struct {
  /* From option2 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option3 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */
} face_mesh_data;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fm_init (void **pdata)
{
  face_mesh_data *data;

  data = *pdata = g_new0 (face_mesh_data, 1);
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
  face_mesh_data *data = *pdata;
  // TODO: free inner data
  UNUSED (data);

  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
fm_setOption (void **pdata, int opNum, const char *param)
{
  face_mesh_data *data = *pdata;

  if (opNum == 0) {
    /* option1 = face mesh decoding Mode */
    // TODO
  } else if (opNum == 1) {
    /* option2 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR ("mode-option-2 of facemesh is output video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE; /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING ("mode-option-2 of facemesh is output video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
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
      GST_ERROR ("mode-option-3 of facemesh is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE; /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING ("mode-option-3 of facemesh is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->i_width = dim[0];
    data->i_height = dim[1];
    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
fm_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  face_mesh_data *data = *pdata;
  GstCaps *caps;
  char *str;

  // TODO: check the input tensor structure (config)
  g_return_val_if_fail (config != NULL, NULL);
  g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

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

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
fm_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  face_mesh_data *data = *pdata;
  const size_t size = (size_t) data->width * data->height * 4; /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  gboolean need_output_alloc;

  UNUSED (config);
  UNUSED (input);

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  if (need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-face_mesh.\n");
    goto error_free;
  }

  memset (out_info.data, 0, size);
  // XXX: for test
  memset (out_info.data, 0xFF0000FF, size / 2);
  // TODO: draw face mesh

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

static gchar decoder_subplugin_face_mesh[] = "face_mesh";

/** @brief Face Mesh tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef faceMesh = { .modename = decoder_subplugin_face_mesh,
  .init = fm_init,
  .exit = fm_exit,
  .setOption = fm_setOption,
  .getOutCaps = fm_getOutCaps,
  .getTransformSize = fm_getTransformSize,
  .decode = fm_decode };

/** @brief Initialize this object for tensordec-plugin */
void
init_fm (void)
{
  nnstreamer_decoder_probe (&faceMesh);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_fm (void)
{
  nnstreamer_decoder_exit (faceMesh.modename);
}
