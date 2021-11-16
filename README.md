# YOLOX-DLStreamer
Inference YOLOX model by DLStreamer with display or print results to JSON file

After install Ubuntu18.04/20.04 on your device, please install OpenVINO™ 2021.4.582 Toolkit.
https://docs.openvino.ai/2021.2/openvino_docs_install_guides_installing_openvino_linux.html

Next execute the following commands to run YOLOX DLStreamer pipeline.
$ cd $workspace
$ git clone https://github.com/Megvii-BaseDetection/YOLOX.git
$ cd YOLOX

Download yolox_display.py or yolox_saveJSON.py from https://github.com/biyuehuang/YOLOX-DLStreamer 
yolox_display.py will show media analytics pipeline on display. 
yolox_saveJSON.py will save the results of YOLOX to JSON file without display.
$ cp yolox_ saveJSON.py $workspace/YOLOX
$ cp yolox_display.py $workspace/YOLOX 
$ source /opt/intel/openvino_2021/bin/setupvars.sh
$ python3 yolox_saveJSON.py -i video.mp4 -d yolox_s.xml
$ python3 yolox_display.py -i video.mp4 -d yolox_s.xml
Note: yolox_s.xml is OpenVINO™ model.

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance. You can download pre-trained Pytorch model from https://github.com/Megvii-BaseDetection/YOLOX
For demo in this document, please download OpenVINO™ model directly. For example, download YOLOX-S model from https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python 
