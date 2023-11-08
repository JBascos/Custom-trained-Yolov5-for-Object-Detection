import os
import cv2
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from numpy import random
from models.experimental import attempt_load
from utils.plots import plot_one_box
from utils.datasets import LoadStreams, LoadImages
from utils.general import scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier

#Engine
import pyttsx3
from utils.vi_hat import main_req
import requests
import threading

#Bootup Sound
engine = pyttsx3.init()
voice = engine.getProperty('voices')  # get the available voices
engine.setProperty('voice', voice[1].id)
engine.say("Hello!")
time.sleep(1)
engine.say("I am Intelligent Sight.")
time.sleep(1)
engine.say("I am ready to help you.")
time.sleep(2)
engine.say("Initializing Detection.")
time.sleep(3)

os.environ['OPENBLAS_NUM_THREADS'] = '1'

#perspective projection constants
person_width_px = 670.0
focal_person = 1675.0
dog_width_px = 473.0
focal_dog = 909.6153564453125
cat_width_px = 473.0
focal_cat = 1689.2857666015625
car_width_px = 651.0
focal_car = 3255.0
bicycle_width_px = 436.0
focal_bicycle = 3633.333251953125

class main:
    def __init__(self):
        self.opt = argparse.Namespace(
            weights='weights/engenium_yolov5.pt',
            source='sample', img_size=640,
            conf_thres=0.2,iou_thres=0.2, device='',
            view_img=False, save_txt=False,
            save_conf=False, nosave=False,
            classes=None, agnostic_nms=False,
            augment=False, update=False,
            project='runs/detect', name='exp',
            exist_ok=False, read=False,
        )
        
        #for distance estimation
        self.img_wth = 0
        self.label = ''
        self.const_dist = 25.0
        self.CONFIDENCE_THRESHOLD = 0.4
        self.NMS_THRESHOLD = 0.3
        self.distance = 0
        
    def engine_alert(self, str, engine):
        if not engine._inLoop:
            engine.say(str)
            engine.runAndWait()

    def config(self, weights, source, classes, read, view_img):
        self.opt.weights = weights
        self.opt.source = source
        self.opt.classes = classes
        self.opt.read = read
        self.opt.view_img = view_img
        
    def projection(self, focal_length, img_wth):
        distance = (focal_length * self.const_dist) / img_wth
        distance = distance / 100
        return distance

    def detect(self, save_img=False):
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        save_img = not self.opt.nosave and not source.endswith(
            '.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        save_dir = Path(increment_path(Path(self.opt.project) /
                        self.opt.name, exist_ok=self.opt.exist_ok))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  
        model = attempt_load(weights, map_location=device)  
        stride = int(model.stride.max())  
        imgsz = check_img_size(imgsz, s=stride)
        if half:
            model.half()  
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True 
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time_synchronized()
            pred = model(img, augment=self.opt.augment)[0]
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            for i, det in enumerate(pred): 
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name) 
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:] 
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    detected_classes = []
                    detected_distance = []
                    detected_area = []
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (
                                cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # get the width and height of the bounding box
                            self.img_wth = xyxy[2] - xyxy[0]
                            # Get the x-coordinate of the center of the bounding box
                            bbox_center = (xyxy[0] + xyxy[2]) / 2
                            # Get the x-coordinate of the center of the image
                            image_center = im0.shape[1] / 2

                            # Determine if the bounding box is on the left, center, or right of the image
                            if bbox_center < image_center - 50:
                                detected_area.append("left")
                            elif bbox_center > image_center + 50:
                                detected_area.append("right")
                            else:
                                detected_area.append("center")

                            self.label = f'{names[int(cls)]} {int(cls)}'

                            if (self.opt.read == False):

                                if names[int(cls)] == 'person':
                                    self.distance = self.projection(focal_person, self.img_wth)
                                elif names[int(cls)] == 'dog':
                                    self.distance = self.projection(focal_dog, self.img_wth)
                                elif names[int(cls)] == 'cat':
                                    self.distance = self.projection(focal_cat, self.img_wth)
                                elif names[int(cls)] == 'car':
                                    self.distance = self.projection(focal_car, self.img_wth)
                                elif names[int(cls)] == 'bicycle':
                                    self.distance = self.projection(focal_bicycle, self.img_wth)
                                
                                if self.distance < 6:
                                    # set colors to red
                                    if self.distance < 2:
                                        detected_classes.append(names[int(cls)])
                                        detected_distance.append(self.distance)
                                        
                                        label = f'{names[int(cls)]} {conf:.2f} {self.distance:.2f} m'
                                        colors[int(cls)] = [0, 0, 255]
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                    else:
                                        detected_classes.append(names[int(cls)])
                                        detected_distance.append(self.distance)
                                        label = f'{names[int(cls)]} {conf:.2f} {self.distance:.2f} m'
                                        colors[int(cls)] = [0, 255, 0]
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            
                    #Auditory Output
                    str_dist = [f"close" if distance < 2 else f"{round(float(distance), 1)} meters away" for distance in detected_distance]
                    str_pos = ["on left" if detected_area[i] == 'left' else "in front" if detected_area[i] == 'center' else "on right" for i in range(len(detected_area))]

                    str_whole = ", ".join([f"{clazz} {str_dist[i]} {str_pos[i]}" for i, clazz in enumerate(detected_classes)])
                    if len(str_whole) > 0: 
                        for position in detected_area:
                            if position == "left":
                                print("Left")
                                # requests.get("http://192.168.4.4/left/active")
                                # requests.get("http://192.168.4.4/left/active")
                            elif position == "right":
                                print("Right")
                                # requests.get("http://192.168.4.1/right/active")
                                # requests.get("http://192.168.4.1/right/active")
                            else:
                                print("Center")
                                # main_req()
                        if len(detected_classes) > 0:
                            speech = f"Detected a {str_whole}"
                            print(f'Speech: {speech}')
                            # Start a new thread to run the engine alert function                    
                            tts_thread = threading.Thread(target=self.engine_alert, args=(speech,engine))
                            tts_thread.start()
                    
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                key = cv2.waitKey(1)
                if key == ord('q'):
                    engine.stop()
                    break
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        
def start():
    run = main()
    global focal_person, focal_dog, focal_cat, focal_car, focal_bicycle
    run.config('weights/engenium_yolov5.pt',0, None, False, True)
    run.detect()

start()
