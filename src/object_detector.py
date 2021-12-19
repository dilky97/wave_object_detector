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

class ObjectDetector:

    def __init__(self) -> None:
        self.cfg = get_cfg()
        # Force model to operate within CPU, erase if CUDA compatible devices ara available
        self.cfg.MODEL.DEVICE = 'cpu'
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(self.cfg)

    def __visualize_output(self, img, outputs):
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        pil_img = Image.fromarray(out.get_image()[:, :, ::-1])
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        buff.seek(0)
        img_str = base64.b64encode(buff.read()).decode("utf-8")

        return img_str

    # def discriminate(self, outputs):
    #     pred_classes = np.array(outputs['instances'].pred_classes)
    #     mask = np.isin(pred_classes, CLASSES)
    #     idx = np.nonzero(mask)
        
    #     # Get Instance values as a dict and leave only the desired ones
    #     out_fields = outputs['instances'].get_fields()
    #     for field in out_fields:
    #         out_fields[field] = out_fields[field][idx]

    #     return outputs


    def detect_objects(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)

        # self.cfg, predictor = load_predictor()
        outputs = self.predictor(img)
        result_img = self.__visualize_output(img, outputs)

        assert (type(result_img)==str), "Output needs to be a String" 

        return result_img

def main():
    predict = ObjectDetector().detect_objects('./static/nyc.jpg')

    im_bytes = base64.b64decode(predict)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  
    img = Image.open(im_file)
    img.show()


if __name__ == "__main__":
    main()

