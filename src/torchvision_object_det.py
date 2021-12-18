import torch
from torchvision_object_det import models, transforms
import numpy as np
import cv2
# from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# def load_pretrained_model():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#     # model_path = './models/cnn14.pth'
#     # if not os.path.exists(model_path):
#     #     print('Downloading model')
#     #     urllib.request.urlretrieve(MODEL_URI, model_path)
#     model.to(device)
#     _ = model.eval()
#     return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model_path = './models/cnn14.pth'
# if not os.path.exists(model_path):
#     print('Downloading model')
#     urllib.request.urlretrieve(MODEL_URI, model_path)
model.to(device)
model.eval()

def detect_objects(img, threshold, CATEGORY_NAMES,device='cpu'):

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)[0]
    
    treshold_mask = (pred['scores'] > threshold)
    boxes = torch.reshape(torch.masked_select(pred['boxes'], treshold_mask.unsqueeze(1)), (-1, 4)).type(torch.int32).cpu().numpy()
    labels = torch.masked_select(pred['labels'], treshold_mask).cpu().numpy()
    labels = [CATEGORY_NAMES[i] for i in labels]
    
    return boxes, labels

def draw_boxes(img, threshold, CATEGORY_NAMES, device='cpu'):
    boxes, labels = detect_objects(img, threshold, CATEGORY_NAMES, device)
    
    if not len(boxes): # draw nothing if no objects are detected
        return img
    
    for box, label in zip(boxes, labels):
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=3)
        img = cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=3)
        
    return img


# odj_model = load_pretrained_model()
image_path = 'C:/Users/dilky/Pictures/Screenshots/Screenshot(155).png'
# PIL_image = Image.open(image_path).convert('RGB')

PIL_image = cv2.imread(image_path)
PIL_image = draw_boxes(
                img = PIL_image,
                threshold=0.7,
                CATEGORY_NAMES=COCO_INSTANCE_CATEGORY_NAMES)

PIL_image.show()