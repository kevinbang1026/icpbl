import os
import time
import traceback
from multiprocessing.managers import BaseManager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.multiprocessing import Manager, Pool

from sort.sort import Sort
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  scale_boxes, xyxy2xywh)
from yolov5.utils.parser import get_config
from yolov5.utils.plots import Annotator, colors


### YOLO INSTANCE ###
class YOL():
    DRAW = True
    SAVE = True
    WEIGHT = 'yolov5m6'
    PREGRAB = 3
    MULTIGPU = False

    OPCODE_DETECTION = 1
    OPCODE_STATUS = 2
    OPCODE_RESERVED = 3
    OPCODE_CLIENT_AUTH = 4
    OPCODE_WATCHDOG_FRED = 5

    CLASS_NAMES = None

    TOTAL_SRC_COUNT = 0
    NORMAL_SRC_COUNT = 0
    SRC_LIST = dict()

    def __init__(self):
        cudnn.benchmark = True
        
        sources = get_config()
        sources.merge_from_file('./sources.yaml')
        
        self.OPENCV_THREADS = sources['OPENCV_THREADS']
        
        self.WEIGHT = sources['WEIGHT']
        self.DRAW = sources['DRAW_ENABLED']
        self.SAVE = sources['SAVE_ENABLED']
        self.PREGRAB = sources['PREGRAB']
        self.MULTIGPU = sources['MULTIGPU_ENABLED']
        self.SCALE = sources['SCALE']

        for src in sources['SOURCES']:
            try:
                self.SRC_LIST[src.split('SRC')[1]] = sources['SOURCES'][src]
            
            except:
                continue
        
        self.TOTAL_SRC_COUNT = len(self.SRC_LIST)

    def get(self):
        return self

class MyManager(BaseManager):
    pass

MyManager.register('YOL', YOL)

class Detection():
    objectDict = None

    def __init__(self, x1, x2, y1, y2, type, id, prob):
        self.objectDict = dict()
        self.objectDict['x1'] = x1
        self.objectDict['x2'] = x2
        self.objectDict['y1'] = y1
        self.objectDict['y2'] = y2
        self.objectDict['objectType'] = type
        self.objectDict['identification'] = id
        self.objectDict['probability'] = prob
    
    def getDict(self):
        return self.objectDict


###############
### Capture ###
###############
def detection(source_url, source_id, PREGRAB, DRAW, SOURCE_STATUS, FRAME_STATUS, FRAME_HOLDER, FRAME_RATE):
    while True:
        captureForever(source_url, source_id, PREGRAB, DRAW, SOURCE_STATUS, FRAME_STATUS, FRAME_HOLDER, FRAME_RATE)
            
def captureForever(source_url, source_id, PREGRAB, DRAW, SOURCE_STATUS, FRAME_STATUS, FRAME_HOLDER, FRAME_RATE):
    try:
        source_id_string = str(source_id)

        cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
        FRAME_RATE[source_id] = cap.get(cv2.CAP_PROP_FPS)
        FRAME_STATUS[source_id] = time.time()

        while cap.isOpened():
            now = time.time()

            if FRAME_HOLDER[source_id] is not None:
                time.sleep(0.02)
                frame = None

                for i in range (0, PREGRAB):
                    ret = cap.grab()

                    if not ret:
                        break

                    ret, frame = cap.retrieve()
                    
                continue

            time.sleep(0.033)
            if now - FRAME_STATUS[source_id] > 3:
                raise Exception('No frame was captured!')

            for i in range (0, PREGRAB):
                ret = cap.grab()

                if not ret:
                    break

                ret, frame = cap.retrieve()

            if ret:
                if now - SOURCE_STATUS[source_id] > 15:
                    SOURCE_STATUS[source_id] = now
                FRAME_STATUS[source_id] = now
                FRAME_HOLDER[source_id] = frame
            
    except Exception as e:
        print("Some errors were occured while detecting thread running!")
        traceback.print_exc()
    
    if DRAW:
        try:  
            cv2.destroyWindow(source_id_string)
        except:
            print("Destroy " + source_id_string + " Failure")

    cap.release()
    time.sleep(2)


#################
### Inference ###
#################
def infer(source_id, yolo_model, cudaDevice, sort_tracker, YOLO_INSTANCE, FRAME_HOLDER, TRAFFIC):
    source_id_string = str(source_id)
    torch.set_float32_matmul_precision('high')
    scale = YOLO_INSTANCE.SCALE
    yolo_model = yolo_model.to(cudaDevice)

    cars = dict()

    while True:
        start = time.time()

        if YOLO_INSTANCE.DRAW or YOLO_INSTANCE.SAVE: 
            cv2.waitKey(1)
        else:
            time.sleep(0.001)

        try:
            if FRAME_HOLDER[source_id] is not None:
                with torch.no_grad():
                    SIZE = (640, 360)
                    frame0 = FRAME_HOLDER[source_id]
                    FRAME_HOLDER[source_id] = None
                        
                    frame2 = cv2.resize(frame0.copy(), SIZE, interpolation=cv2.INTER_LINEAR)
                    frame2 = cv2.copyMakeBorder(frame2, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                    frame2 = np.stack([frame2], 0)
                    frame2 = frame2[..., ::-1].transpose((0, 3, 1, 2))
                    frame2 = np.ascontiguousarray(frame2)
                    frame2 = torch.from_numpy(frame2).to(cudaDevice)
                    frame2 = frame2.half()
                    frame2 /= 255.0

                    if frame2.ndimension() == 3:
                        frame2 = frame2.unsqueeze(0)
                    pred = yolo_model(frame2)[0]
                    pred = non_max_suppression(pred, conf_thres=0.5, agnostic=True)

                    if YOLO_INSTANCE.DRAW or YOLO_INSTANCE.SAVE:
                        annotator = Annotator(frame0, line_width=2, pil=not ascii)
                        s = '%g: ' % 0

                    det = pred[0]
                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(frame2.shape[2:], det[:, :4], frame0.shape).round()

                        if YOLO_INSTANCE.DRAW or YOLO_INSTANCE.SAVE:
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()
                                s += f"{n} {YOLO_INSTANCE.CLASS_NAMES[int(c)]}{'s' * (n > 1)}, "

                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]


                        for i, [x, y, w, h] in reversed(list(enumerate(xywhs.tolist()))):
                            if w > SIZE[0]/2:
                                det = torch.cat([det[0:i], det[i+1:]])

                        # Pass detections to SORT
                        # NOTE: We send in detected object class too
                        outputs = np.empty((0,6))
                        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                           outputs = np.vstack((outputs, np.array([x1, y1, x2, y2, conf, detclass]
                           )))

                        # Run SORT
                        outputs = sort_tracker.update(outputs)
                    
                    detections = []
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        boxes = output[0:4]
                        oid = output[4]
                        cls = output[5]
                        c = int(cls)

                        if c > 2:
                            continue
                        
                        if scale == 1:
                            detection = Detection(int(boxes[0]), int(boxes[2]), int(boxes[1]), int(boxes[3]), YOLO_INSTANCE.CLASS_NAMES[c], str(oid), f'{conf:.2f}')
                        else:
                            detection = Detection(round(int(boxes[0]) * scale), round(int(boxes[2]) * scale), round(int(boxes[1]) * scale), round(int(boxes[3]) * scale), YOLO_INSTANCE.CLASS_NAMES[c], str(oid), f'{conf:.2f}')
                        
                        detections.append(detection)

                        if oid in cars:
                            cars[oid][1] = start - cars[oid][0]
                            cars[oid][2] = 0
                            cars[oid][3] = 1
                        else:
                            cars[oid] = [start, 0, 0, 0] # start time, latest time, delay, flag

                        if YOLO_INSTANCE.DRAW or YOLO_INSTANCE.SAVE:
                            label = f'{int(oid)}, {conf:.2f}'
                            annotator.box_label(boxes, label, color=colors(c, True))

                    t_flag = 0

                    for car, info in cars.copy().items():
                        if info[2] >= 5:
                            del cars[car]
                        if info[3] == 0:
                            info[2] += 1
                        else:
                            info[3] = 0
                        
                        if len(detections) > 30 or info[1] > 150:
                            if t_flag < 1:
                                t_flag = 1
                        if len(detections) > 60 or info[1] > 300:
                            if t_flag < 2:
                                t_flag = 2
                        
                    TRAFFIC[source_id] = '원활' if t_flag == 0 else '서행' if t_flag == 1 else '정체'

                    if YOLO_INSTANCE.DRAW or YOLO_INSTANCE.SAVE:
                        im0 = annotator.result()
                        im0 = cv2.resize(im0, (1280, 720), interpolation=cv2.INTER_LINEAR)
                        if YOLO_INSTANCE.DRAW:
                            cv2.imshow(source_id_string, im0)
                        if YOLO_INSTANCE.SAVE:
                            cv2.imwrite('./save/cam' + source_id_string + '/' + str(int(time.time())) + '.png', im0)

        except Exception as e:
            traceback.print_exc()
            continue


def show_traffic(traffic:dict):
    while True:
        print(time.strftime('%c', time.localtime(time.time())), ' ', dict(reversed(traffic.items())), end= '\r')
        time.sleep(1)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    yolo_instance = YOL()
    manager = Manager()
    SOURCE_STATUS = manager.dict()
    FRAME_STATUS = manager.dict()
    FRAME_HOLDER = manager.dict()
    FRAME_RATE = manager.dict()
    TRAFFIC = manager.dict()

    cv2.setNumThreads(yolo_instance.OPENCV_THREADS)

    if not yolo_instance.MULTIGPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_device_count = torch.cuda.device_count()
    print("CUDA Devices: %d" % cuda_device_count)

    cuda_device = torch.device('cuda')
    yolo_model = attempt_load('./yolov5/weights/' + yolo_instance.WEIGHT + '.pt')
    yolo_model(torch.zeros(1, 3, 480, 480)
        .type_as(next(yolo_model.parameters())))
    yolo_model.half()
    check_img_size(480, s=int(yolo_model.stride.max()))
    yolo_model.share_memory()
    yolo_instance.CLASS_NAMES = (yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names)

    now = time.time()
    pool = Pool(yolo_instance.TOTAL_SRC_COUNT*2+1)

    for index, (i, url) in enumerate(yolo_instance.SRC_LIST.items()):
        SOURCE_STATUS[i] = now - 60
        FRAME_HOLDER[i] = None
        pool.apply_async(detection, args=(url, i, yolo_instance.PREGRAB, yolo_instance.DRAW, SOURCE_STATUS, FRAME_STATUS, FRAME_HOLDER, FRAME_RATE,))

        Path('./save/cam' + i).mkdir(exist_ok= True, parents= True)
        sort_tracker = Sort(iou_threshold=0.2, max_age=30, min_hits= 2)
        pool.apply_async(infer, args=(i, yolo_model, cuda_device, sort_tracker, yolo_instance, FRAME_HOLDER, TRAFFIC,))

    show_traffic(traffic= TRAFFIC)