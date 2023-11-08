# Custom-trained-Yolov5-for-Object-Detection

  YOLO is regarded as a famous object detection algorithm that now has multiple versions. YOLO is a 1-stage detector model which means it simultaneously performs regional proposal and classification. 
Backbone is a Convolutional Neural Network (CNN) formed by combining features of images in multiple particle sizes. The neck comprises a series of layers that mix and combine image features to transfer before prediction, and the Head uses the transferred features from the Neck (PAnet) and goes through the box and class prediction process. The remarkable feature of YOLOv5 is the Focus and CSP (cross-stage partial connections) layer.
YOLOv5 uses the same head as YOLOv3 and YOLOv4 as seen in Figure 1. It is composed of three convolution layers that predict the location of the bounding boxes (x, y, height, width), the scores and the objects classes.

![image](https://github.com/JBascos/Custom-trained-Yolov5-for-Object-Detection/assets/150259866/c9370b3f-b1aa-45ca-8fcf-50ad0d2cee07)

  The YOLOv5 algorithm was employed as the model for the obstruction detection system. To train the model, a dataset comprising 8,000 images from the COCO dataset and an additional 6,000 images captured specifically from the vicinity of Bahay Biyaya, Immaculate Concepcion, and Farmer's Cubao in the Philippines will be utilized. This selection of images is recommended by the client, as it represents the areas where the system will be frequently used for navigation. By training the model on this diverse dataset, it is expected to achieve accurate and reliable detection performance in real-world scenarios encountered by the client. 

  After finalizing the algorithm, the wearable assistive device is physically constructed using a fedora hat as the main container. Additionally, wristbands are incorporated into the design, which house the ESP32 microcontroller and coin motors responsible for generating vibratory haptic feedback. The image below provides a visual representation of the wearable assistive device, including the vibratory haptic feedback wristbands.

![image](https://github.com/JBascos/Custom-trained-Yolov5-for-Object-Detection/assets/150259866/af42a28a-85ff-47c2-a56f-f1d456a946cd)![image](https://github.com/JBascos/Custom-trained-Yolov5-for-Object-Detection/assets/150259866/0edd5e6a-7533-4e20-9a09-acea6dabbd9c)



