import unittest
from types import SimpleNamespace
import numpy as np
import torch
from cafe_ml.camera import CameraPipeline,draw
from cafe_ml.detector import yolo_box

class FixedClassifier(torch.nn.Module):
    def forward(self,x):
        return torch.tensor([[0.,0.,8.,0.,0.]]).repeat(len(x),1)

class CameraTests(unittest.TestCase):
    def pipeline(self,manual=False):
        p=CameraPipeline.__new__(CameraPipeline)
        p.device='cpu';p.manual=manual;p.detector_conf=.3;p.classifier_conf=.55
        def transform(image):
            self.assertEqual(image.getpixel((0,0)),(255,0,0))
            return torch.zeros(3,224,224)
        p.models=[('test',FixedClassifier(),transform)]
        boxes=SimpleNamespace(xyxy=torch.tensor([[10.,12.,45.,48.]]),conf=torch.tensor([.9]))
        p.detector=SimpleNamespace(predict=lambda *a,**kw:[SimpleNamespace(boxes=boxes)])
        return p

    def test_localize_crop_rgb_classify_draw(self):
        frame=np.zeros((80,100,3),dtype=np.uint8);frame[:,:,2]=255
        annotations=self.pipeline().process(frame)
        self.assertEqual(annotations[0]['bbox'],[10,12,45,48])
        self.assertEqual(annotations[0]['class'],'ripe')
        before=frame.copy();output=draw(frame,annotations)
        self.assertTrue(np.array_equal(frame,before))
        self.assertFalse(np.array_equal(output,frame))

    def test_low_score_is_unknown_and_manual_roi_is_explicit(self):
        p=self.pipeline(manual=True);p.classifier_conf=1.
        frame=np.zeros((80,100,3),dtype=np.uint8);frame[:,:,2]=255
        result=p.process(frame,(0,0,20,30))
        self.assertEqual(result[0]['class'],'unknown')
        self.assertIsNone(result[0]['detector_score'])
        self.assertEqual(result[0]['bbox'],[0,0,20,30])

    def test_yolo_conversion_clips_to_image(self):
        np.testing.assert_allclose(yolo_box([-10,-5,40,25],100,80),[.15,.125,.3,.25])
        with self.assertRaises(ValueError): yolo_box([200,0,10,10],100,80)

    def test_no_detection_does_not_classify_background(self):
        p=self.pipeline()
        empty=SimpleNamespace(xyxy=torch.empty((0,4)),conf=torch.empty(0))
        p.detector=SimpleNamespace(predict=lambda *a,**kw:[SimpleNamespace(boxes=empty)])
        self.assertEqual(p.process(np.zeros((80,100,3),dtype=np.uint8)),[])

if __name__=='__main__':unittest.main()
