import numpy as np
from PIL import Image
import base64
from io import BytesIO

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

CLASSES = [0, 17]

def load_predictor():
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


# def inference(cfg, img):
#     predictor = DefaultPredictor(cfg)
#     return predictor(img)


def visualize_output(cfg, img, outputs):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    pil_img = Image.fromarray(out.get_image()[:, :, ::-1])
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    buff.seek(0)
    img_str = base64.b64encode(buff.read()).decode("utf-8")

    return img_str


def discriminate(outputs):
    pred_classes = np.array(outputs['instances'].pred_classes)
    mask = np.isin(pred_classes, CLASSES)
    idx = np.nonzero(mask)
    
    # Get Instance values as a dict and leave only the desired ones
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

    return outputs

def detect_objects(image_path,cfg,predictor):
    # img = cv2.imread(image_path)
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # cfg, predictor = load_predictor()
    outputs = predictor(img)
    result_img = visualize_output(cfg, img, outputs)

    assert (type(result_img)==str), "Output needs to be a String" 

    return result_img


# def main():
#     #img = cv2.imread('img.png')
#     # img_path = cv2.imread('C:/Users/dilky/Pictures/Screenshots/Screenshot(155).png')
#     img_path = './data/f/c9c6b133-c4fb-461d-8d27-9b4140515b05/nyc.jpg'

#     # cv2.imshow('kk', img)
#     # cv2.waitKey(0)
#     # cfg = load_cfg()
#     # cfg, predictor = load_predictor()
#     outputs = detect_objects(img_path)
#     # aaa = outputs.copy()
#     # bbb = discriminate(outputs)
#     # out = visualize_output(cfg, img, outputs)
#     cv2.imshow('kk', outputs)
#     cv2.waitKey(0)


# if __name__ == "__main__":
#     main()