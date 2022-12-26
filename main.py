import glob
import io
import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import wget
from torchvision.transforms import Compose, ToTensor

from model import decoder, encoder

WEIGHT_PATH = './weights/best_weight2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(object):
    def __init__(self) -> None:
        self.model_Enc = encoder.Encoder_RRDB(num_feat=64).to(device=DEVICE)
        self.model_Dec_SR = decoder.Decoder_SR_RRDB(num_in_ch=64).to(device=DEVICE)
        self.preprocess = Compose([ToTensor()])
        self.load_model()

    def load_model(self, weight_path=WEIGHT_PATH):
        if not os.path.isfile("./weights/best_weight2.pth"):
            response = wget.download("https://raw.githubusercontent.com/hungnguyen2611/super-resolution/master/weights/best_weight.pth", "./weights/best_weight.pth")
        weight = torch.load(weight_path)
        print("[LOADING] Loading encoder...")
        self.model_Enc.load_state_dict(weight['model_Enc'])
        print("[LOADING] Loading decoder...")
        self.model_Dec_SR.load_state_dict(weight['model_Dec_SR'])
        print("[LOADING] Loading done!")
        self.model_Enc.eval()
        self.model_Dec_SR.eval()

    def predict(self, img):
        with torch.no_grad():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.preprocess(img)
            img = img.unsqueeze(0)
            img = img.to(DEVICE)

            feat = self.model_Enc(img)
            out = self.model_Dec_SR(feat)
            min_max = (0, 1)
            out = out.detach()[0].float().cpu()

            out = out.squeeze().float().cpu().clamp_(*min_max)
            out = (out - min_max[0]) / (min_max[1] - min_max[0])
            out = out.numpy()
            out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))

            out = (out*255.0).round()
            out = out.astype(np.uint8)
            return out

model = Model()

def predict(img):
    global model
    img.save("test/1.png", "PNG")
    image = cv2.imread("test/1.png", cv2.IMREAD_COLOR)
    if image.shape[0] > 300 or image.shape[1] > 300:
        raise gr.Error(
            f'Image is too large. Please uploade <= 300 px img')
    out = model.predict(img=image)

    cv2.imwrite(f'images_uploaded/1.png', out)
    return f"images_uploaded/1.png"




if __name__ == '__main__':
    title = "Super-Resolution Demo USR-DA Unofficial 🚀🚀🔥"
    description = ''' 
<br>
**This Demo expects low-quality and low-resolution images**
**We are looking for collaborators! Collaborator** 
</br>
'''
    article = "<p style='text-align: center'><a href='https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.pdf' target='_blank'>Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective</a> | <a href='https://github.com/hungnguyen2611/super-resolution.git' target='_blank'>Github Repo</a></p>"
    examples= glob.glob("testsets/*.png")
    gr.Interface(
        predict, 
        gr.inputs.Image(type="pil", label="Input").style(height=260),
        gr.inputs.Image(type="pil", label="Ouput").style(height=240),
        title=title,
        description=description,
        article=article,
        examples=examples,
        ).launch(enable_queue=True, share=True)




    



