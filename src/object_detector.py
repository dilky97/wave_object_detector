import numpy as np
from PIL import Image
import base64
from io import BytesIO

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class ObjectDetector:

    """
    Program to detect objects in an image using pretrained Faster RCNN network in Detectron2 Library.
    Adapted from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    """

    def __init__(self) -> None:
        self.cfg = get_cfg()
        # Force model to operate within CPU, erase if CUDA compatible devices ara available
        self.cfg.MODEL.DEVICE = 'cpu'
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        self.classes_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

        self.predictor = DefaultPredictor(self.cfg)

    def __visualize_output(self, img, outputs):
        # Use `Visualizer` to draw the predictions on the image.
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Get base64 encoded image
        pil_img = Image.fromarray(out.get_image()[:, :, ::-1])
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        buff.seek(0)
        img_str = base64.b64encode(buff.read()).decode("utf-8")

        return img_str

    def __pick_classes(self, outputs, picked_classes):
        """
        Filter the chosen classes from the predictions
        :param outputs: predictions returned by the model inference
        :param picked_classes: user picked values from the picker

        :returns: filtered outputs
        """

        pred_classes = np.array(outputs['instances'].pred_classes)
        mask = np.isin(pred_classes, picked_classes)
        idx = np.nonzero(mask)
        
        # Get the picked classes from predicted Instances
        out_fields = outputs['instances'].get_fields()
        for field in out_fields:
            out_fields[field] = out_fields[field][idx]

        return outputs


    def detect_objects(self, image_path, picked_classes):
        """
        Detect objects in a supplied image
        :param image_path: Path to the image file
        :param picked_classes: user picked values from the picker

        :returns: Annotated image with bounding box detections
        """

        img = Image.open(image_path).convert('RGB')
        img = np.array(img)

        outputs = self.predictor(img)
        if not picked_classes:
            result_img = self.__visualize_output(img, outputs)
        else:
            mask = np.isin(self.classes_names, picked_classes)
            class_idxs = np.nonzero(mask)
            picked_outputs = self.__pick_classes(outputs, class_idxs)
            result_img = self.__visualize_output(img, picked_outputs)

        assert (type(result_img)==str), "Output needs to be a String" 

        return result_img

def main():
    detector = ObjectDetector()
    predict = detector.detect_objects('./static/nyc.jpg', ['person'])

    im_bytes = base64.b64decode(predict)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  
    img = Image.open(im_file)
    img.show()


if __name__ == "__main__":
    main()

