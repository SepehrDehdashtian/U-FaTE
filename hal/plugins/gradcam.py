import torch
import os
# from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


__all__ = ['GradCam']


class GradCam:
    def __init__(self, 
                 opts,
                 model, 
                 target_layer = None,
                 ):
        
        self.opts = opts
        self.model = model
        if target_layer is None:
            # Set the last layer as target layer
            pass

        import pdb; pdb.set_trace()
        self.cam = GradCAM(
                        model=model,
                        # target_layer=model.layer4[-1],
                        # preprocess_function=preprocess,
                        use_cuda=False
                        )
        
        self.cam.batch_size = 16

    def calculate(self, image):
        explanation = self.explainer.explain(image)
        return explanation

    def save(self, path, img_name):
        tau = self.opts.tau
        filename = os.path.join(path, f'{img_name}_GradCam-tau_{tau}.jpeg')
        pass


    def explain_group(self, images):
        for img in images:
            exp = self.calculate(img)
            dir(exp)
            import pdb; pdb.set_trace()


class GradCam00:
    def __init__(self, 
                 opts,
                 model, 
                 preprocess,
                 target_layer = None,
                 ):
        
        self.opts = opts
        self.model = model
        if target_layer is None:
            # Set the last layer as target layer
            pass

        import pdb; pdb.set_trace()
        self.explainer = GradCAM(
                        model=model,
                        target_layer=model.layer4[-1],
                        preprocess_function=preprocess
                        )

    def calculate(self, image):
        explanation = self.explainer.explain(image)
        return explanation

    def save(self, path, img_name):
        tau = self.opts.tau
        filename = os.path.join(path, f'{img_name}_GradCam-tau_{tau}.jpeg')
        pass


    def explain_group(self, images):
        for img in images:
            exp = self.calculate(img)
            dir(exp)
            import pdb; pdb.set_trace()

