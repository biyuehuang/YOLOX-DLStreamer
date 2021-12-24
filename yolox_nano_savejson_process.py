import sys
from argparse import ArgumentParser
import gi  # get Python bindings for GLib-based libraries
gi.require_version('GstVideo', '1.0')
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GstVideo, GObject

# GVA API modules
from gstgva import VideoFrame, util
import numpy as np
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess

parser = ArgumentParser(add_help=False)
_args = parser.add_argument_group('Options')
_args.add_argument("-i", "--input", help="Required. Path to input video file",
                   required=True, type=str)
_args.add_argument("-d", "--detection_model", help="Required. Path to an .xml file with object detection model",
                   required=True, type=str)

# init GStreamer
Gst.init(sys.argv)

# post-processing code    
def process_frame(frame: VideoFrame, threshold: float = 0.5) -> bool:
    width = frame.video_info().width
    height = frame.video_info().height

    model_input_size = 416  # 640x640 is yolox input size.
    ratio = min(model_input_size / height, model_input_size / width)
   # offset_x = 0
   # offset_y = 0
    offset_x = (model_input_size - ratio * width) / 2
    offset_y = (model_input_size - ratio * height) / 2

    for tensor in frame.tensors():
        data = tensor.data()
        dims = tensor.dims()
        np_data = data.reshape(dims)

        predictions = demo_postprocess(
            np_data, (model_input_size, model_input_size), p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]

            for k in range(len(final_cls_inds)):
                if final_scores[k] < threshold:
                    continue

                def restore_coordinates(value, ratio, offset):
                    return int((value - offset) / ratio + 0.5)

                x_min = restore_coordinates(final_boxes[k, 0], ratio, offset_x)
                y_min = restore_coordinates(final_boxes[k, 1], ratio, offset_y)
                x_max = restore_coordinates(final_boxes[k, 2], ratio, offset_x)
                y_max = restore_coordinates(final_boxes[k, 3], ratio, offset_y)

                frame.add_region(x_min, y_min, x_max - x_min, y_max - y_min,
                                 COCO_CLASSES[int(final_cls_inds[k])], final_scores[k])

    return True

def detect_postproc_callback(pad, info):
    with util.GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
        caps = pad.get_current_caps()
        frame = VideoFrame(buffer, caps=caps)
        status = process_frame(frame)
    return Gst.PadProbeReturn.OK if status else Gst.PadProbeReturn.DROP

def main():
    args = parser.parse_args()

    # build pipeline using parse_launch
    pipeline_str = "filesrc location={} ! decodebin ! videoconvert ! " \
        "gvainference name=gvainference model={} device=GPU pre-process-backend=opencv batch_size=1 ! queue ! " \
        "gvametaconvert format=json ! gvametapublish method=file file-path=./result ! fakesink".format(
            args.input, args.detection_model)
    pipeline = Gst.parse_launch(pipeline_str)

  
    # set callback
    gvainference = pipeline.get_by_name("gvainference")
    if gvainference:
        pad = gvainference.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, detect_postproc_callback)

    # start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # wait until EOS or error
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

    # free pipeline
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main() or 0)

