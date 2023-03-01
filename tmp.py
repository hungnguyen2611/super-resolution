import torch

ckpt = torch.load("./weights/epoch_4200.pth")

ckpt = {
    "model_Enc": {k.replace("module.", ""): ckpt["model_Enc"][k] for k in ckpt["model_Enc"].keys()}, 
    "model_Dec_SR": {k.replace("module.", ""): ckpt["model_Dec_SR"][k] for k in ckpt["model_Dec_SR"].keys()}
}


torch.save(ckpt,'./weights/best_weight2.pth')