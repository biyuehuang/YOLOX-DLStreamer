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
    
    ratio = 1

    for tensor in frame.tensors():
        dims = tensor.dims()
        data = tensor.data()
        object_size = dims[-1]
        
        np_data = data.reshape(1,8400,85)

        predictions = demo_postprocess(np_data, (height, width), p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]
        
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        
        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]        
            
            for k in range(len(final_cls_inds)):
                if final_scores[k] < threshold:
                    continue
                x_min = int(final_boxes[k,0] + 0.5)
                y_min = int(final_boxes[k,1] + 0.5)
                x_max = int(final_boxes[k,2] + 0.5)
                y_max = int(final_boxes[k,3] + 0.5)
                frame.add_region(x_min, y_min, x_max - x_min, y_max - y_min, COCO_CLASSES[int(final_cls_inds[k])], final_scores[k])

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
    pipeline_str = "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=BGRx,width=640,height=640 ! " \
        "gvainference name=gvainference model={} ! " \
        "gvawatermark ! videoconvert ! fpsdisplaysink video-sink=xvimagesink sync=false".format(
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

