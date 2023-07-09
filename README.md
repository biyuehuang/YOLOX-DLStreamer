# YOLOX-DLStreamer
Inference YOLOX model by DLStreamer with display or print results to JSON file

After install Ubuntu18.04/20.04 on your device, please install OpenVINO™ 2021.4.582 Toolkit.
https://docs.openvino.ai/2021.2/openvino_docs_install_guides_installing_openvino_linux.html

Next execute the following commands to run YOLOX DLStreamer pipeline.
```
$ cd $workspace
$ git clone https://github.com/Megvii-BaseDetection/YOLOX.git
$ cd YOLOX
```
Download yolox_display.py or yolox_saveJSON.py from https://github.com/biyuehuang/YOLOX-DLStreamer 

yolox_display.py will show media analytics pipeline on display. 

yolox_saveJSON.py will save the results of YOLOX to JSON file without display.
```
$ cp yolox_ saveJSON.py $workspace/YOLOX
$ cp yolox_display.py $workspace/YOLOX 
$ source /opt/intel/openvino_2021/bin/setupvars.sh
$ python3 yolox_saveJSON.py -i video.mp4 -d yolox_s.xml
$ python3 yolox_display.py -i video.mp4 -d yolox_s.xml
```
Note: yolox_s.xml is OpenVINO™ model.

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance. You can download pre-trained Pytorch model from https://github.com/Megvii-BaseDetection/YOLOX

For demo in this document, please download OpenVINO™ model directly. For example, download YOLOX-S model from https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python 

## 1	Custom Post-processing
Intel® DL Streamer gvadetect element performs object detection using SSD-like (including MobileNet-V1/V2 and ResNet), YoloV2/YoloV3/YoloV2-tiny/YoloV3-tiny and FasterRCNN-like object detection models.
YOLOX is an object detection model which is an anchor-free version of YOLO. The post-processing of YOLOX is difference from YOLOV2/ YOLOV3. Therefore, Users need to custom post-processing of YOLOX model.

### 1.1	Set output_postproc of Model-Proc File
Intel® DL Streamer Model-Proc-File#output-postproc provide some post-processing converters for object detection and classification. It supports post-processing converters for GVA element such as gvadetect, gvaclassify, gvaaudiodetect, and gvainference, please refer to https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Model-Proc-File#output-postproc-configuration. 

Example of what output_postproc and its parameters can look like are in Model-Proc JSON file, please refer to https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Model-Proc-File#output-postproc-example and https://github.com/openvinotoolkit/dlstreamer_gst/tree/master/samples/model_proc

### 1.2	Set Python Callback in the Middle of GStreamer Pipeline
If post-processing converters are out of Section 3.1 solution, users can consider custom post-processing according to this tutorial: https://github.com/openvinotoolkit/dlstreamer_gst/blob/master/docs/index.md#custom-post-processing-tutorial-
Note that this tutorial uses the gvainference element without specifying the model-proc file and call the raw array for users to post-process the raw array. The raw array is the output of AI model.
This section shows us how to custom post-processing of model i.e., YOLOX. Users can download YOLOX OpenVINO™ model and OpenVINO™ Python demo code from https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python
In order to enable YOLOX in Intel® DL Streamer, users need to modify function process_frame(frame: VideoFrame, threshold: float = 0.5) -> bool in above tutorial. Refer to the post-processing (Step 8. Process output) of openvino_inference.py, user can define function process_frame() of yolox_display.py like the left of Figure 2.

Figure 1.	Post-processing of YOLOX, the left side is Intel® DL Streamer Python Callback, the right side is OpenVINO™ Python Demo
 ![image](https://github.com/biyuehuang/YOLOX-DLStreamer/assets/31006098/ceebc54d-d7d4-4e1e-a319-7edd4402f1f2)


### 1.3	Insert gvapython Element and Provide Python Callback Function
The modification of Python Callback is the same as Section 3.2. Then please refer the following demo code face_detection_and_classification/postproc_callbacks/ssd_object_detection.py  to package function process_frame() to gvapython element.
Finally make this gvapython element as the post-processing converter of gvainference element, such as face_detection_and_classification.sh 

## 2	Custom Pre-processing
Before feed frame into AI model, some pre-processing need to be done for frame. Intel® DL Streamer GVA elements have default pre-processing without specifying the model-proc file. There are other ways for users to custom pre-processing too.

### 2.1	Set input_postproc of Model-Proc File
Intel® DL Streamer Model-Proc-File#input-preproc-configuration provide some pre-processing keys by OpenCV and VAAPI backend, such as resize, crop, color space, normalize, padding, and others.
Example of what input_postproc and its parameters can look like are in Model-Proc JSON file, please refer to https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Model-Proc-File#input-preproc-example and https://github.com/openvinotoolkit/dlstreamer_gst/tree/master/samples/model_proc

For example, YOLOX has customized pre-processing function preproc(img, input_size, swap=(2, 0, 1)) in https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
```
def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r
```
Function preproc(img, input_size, swap=(2, 0, 1)) has resize source image with aspect-ratio and pad resized image to the input size of YOLOX model. 

Please use the gvainference element with specifying the model-proc file to do pre-processing such as https://github.com/biyuehuang/YOLOX-DLStreamer/blob/main/yolox.json. 
```
 {
  "json_schema_version": "2.0.0",
  "input_preproc": [
    {
      "layer_name": "input",
      "precision": "FP32",
      "format": "image",
      "params": {
        "resize": "aspect-ratio",
        "color_space": "BGR",
        "padding": {
          "stride": 0,
          "fill_value": [
            114.0,
            114.0,
            114.0
          ]
        }
      }
    }
  ],
  "output_postproc": []
}
```
Note: gvainference element has image transpose function, so users don’t need to define in model-proc file. Offset you should use only in case aspect-ratio resize, if it is not, set it 0 in process_frame(frame: VideoFrame, threshold: float = 0.5) -> bool https://github.com/biyuehuang/YOLOX-DLStreamer/blob/main/yolox_savejson_process.py.  
```
# post-processing code
def process_frame(frame: VideoFrame, threshold: float = 0.5) -> bool:
    width = frame.video_info().width
    height = frame.video_info().height

    model_input_size = 640  # 640x640 is yolox input size.
    ratio = min(model_input_size / height, model_input_size / width)
    offset_x = (model_input_size - ratio * width) / 2
offset_y = (model_input_size - ratio * height) / 2
```

### 2.2	Use GStreamer Elements in the Middle of GStreamer Pipeline
If pre-processing keys in Section 4.1 can’t meet users’ demands, users can use GStreamer elements which provide flexible plugins.
For example, GStreamer element videobox is padding function: https://gstreamer.freedesktop.org/documentation/videobox/index.html?gi-language=c 
The following GStreamer Pipeline “video/x-raw,format=BGRx,width=640,height=480” can resize source MP4 file to width=640 and height=480. “videobox right=-128” can pad only the right side of frame. Therefore, the final frame size is width=768 and height=480 on display.
```
$ gst-launch-1.0 filesrc location=cut.mp4 ! decodebin ! videoconvert ! video/x-raw,format=BGRx,width=640,height=480 ! videobox right=-128 ! gvadetect model=face-detection-adas-0001.xml ! gvaclassify model=emotions-recognition-retail-0003.xml model-proc=emotions-recognition-retail-0003.json ! gvawatermark ! xvimagesink sync=false
```
For more GStreamer elements please refer to https://gstreamer.freedesktop.org/documentation/plugins_doc.html?gi-language=c


