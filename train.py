import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.LQGT_dataset import LQGTDataset, LQGTValDataset
from model import decoder, discriminator, encoder
from opt.option import args
from util.utils import (RandCrop, RandHorizontalFlip, RandRotate, ToTensor, RandCrop_pair,
                        VGG19PerceptualLoss)

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

wandb.init(project='SR', config=args)



# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print('using GPU 0')
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()

device = torch.device('cuda')

# make directory for saving weights
if not os.path.exists(args.snap_path):
    os.mkdir(args.snap_path)

print("Loading dataset...")
# load training dataset
train_dataset = LQGTDataset(
    db_path=args.dir_data,
    transform=transforms.Compose([RandCrop(args.patch_size, args.scale), RandHorizontalFlip(), RandRotate(), ToTensor()])
)

val_dataset = LQGTValDataset(
    db_path=args.dir_data,
    transform=transforms.Compose([RandCrop_pair(args.patch_size, args.scale), ToTensor()])
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)


print("Create model")
model_Disc_feat = discriminator.UNetDiscriminator(num_in_ch=args.n_hidden_feats).to(device)
model_Disc_img_LR = discriminator.UNetDiscriminator(num_in_ch=3).to(device)
model_Disc_img_HR = discriminator.UNetDiscriminator(num_in_ch=3).to(device)
# define model (generator)
model_Enc = encoder.Encoder_RRDB(num_feat=args.n_hidden_feats).to(device)
model_Dec_Id = decoder.Decoder_Id_RRDB(num_in_ch=args.n_hidden_feats).to(device)
model_Dec_SR = decoder.Decoder_SR_RRDB(num_in_ch=args.n_hidden_feats).to(device)

# define model (discriminator)

# model_Disc_feat = discriminator.UNetDiscriminator(num_in_ch=64).to(device)
# model_Disc_img_LR = discriminator.UNetDiscriminator(num_in_ch=3).to(device)
# model_Disc_img_HR = discriminator.UNetDiscriminator(num_in_ch=3).to(device)

# wandb logging
wandb.watch(model_Disc_feat)
wandb.watch(model_Disc_img_LR)
wandb.watch(model_Enc)
wandb.watch(model_Dec_Id)
wandb.watch(model_Dec_SR)


print("Define Loss")
# loss
loss_L1 = nn.L1Loss().to(device)
loss_MSE = nn.MSELoss().to(device)
loss_adversarial = nn.BCEWithLogitsLoss().to(device)
loss_percept = VGG19PerceptualLoss().to(device)


print("Define Optimizer")
# optimizer 
params_G = list(model_Enc.parameters()) + list(model_Dec_Id.parameters()) + list(model_Dec_SR.parameters())
optimizer_G = optim.Adam(
    params_G,
    lr=args.lr_G,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)
params_D = list(model_Disc_feat.parameters()) + list(model_Disc_img_LR.parameters()) + list(model_Disc_img_HR.parameters())
optimizer_D = optim.Adam(
    params_D,
    lr=args.lr_D,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)

print("Define Scheduler")
# Scheduler
iter_indices = [args.interval1, args.interval2, args.interval3]
scheduler_G = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_G,
    milestones=iter_indices,
    gamma=0.5
)
scheduler_D = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_D,
    milestones=iter_indices,
    gamma=0.5
)

# print("Data Parallel")
# model_Enc = nn.DataParallel(model_Enc)
# model_Dec_Id = nn.DataParallel(model_Dec_Id)
# model_Dec_SR = nn.DataParallel(model_Dec_SR)

# define model (discriminator)
#model_Disc_feat = nn.DataParallel(model_Disc_feat)
#model_Disc_img_LR = nn.DataParallel(model_Disc_img_LR)
#model_Disc_img_HR = nn.DataParallel(model_Disc_img_HR)

print("Load model weight")
# load model weights & optimzer % scheduler
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)

    model_Enc.load_state_dict(checkpoint['model_Enc'])
    model_Dec_Id.load_state_dict(checkpoint['model_Dec_Id'])
    model_Dec_SR.load_state_dict(checkpoint['model_Dec_SR'])
    model_Disc_feat.load_state_dict(checkpoint['model_Disc_feat'])
    model_Disc_img_LR.load_state_dict(checkpoint['model_Disc_img_LR'])
    model_Disc_img_HR.load_state_dict(checkpoint['model_Disc_img_HR'])

    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    scheduler_D.load_state_dict(checkpoint['scheduler_D'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G'])

    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0


if args.pretrained is not None:
    ckpt = torch.load(args.pretrained)
    ckpt["params"]["conv_first.weight"] = ckpt["params"]["conv_first.weight"][:,0,:,:].expand(64,64,3,3)
    model_Dec_SR.load_state_dict(ckpt["params"])
    






# model_Enc = model_Enc.to(device)
# model_Dec_Id = model_Dec_Id.to(device)
# model_Dec_SR = model_Dec_SR.to(device)

# # define model (discriminator)
# model_Disc_feat = model_Disc_feat.to(device)
# model_Disc_img_LR = model_Disc_img_LR.to(device)
# model_Disc_img_HR =model_Disc_img_HR.to(device)
# training

PSNR = PeakSignalNoiseRatio()
SSIM = StructuralSimilarityIndexMeasure()
LPIPS = LearnedPerceptualImagePatchSimilarity()

for epoch in range(start_epoch, args.epochs):
    # generator
    model_Enc.train()
    model_Dec_Id.train()
    model_Dec_SR.train()

    # discriminator
    model_Disc_feat.train()
    model_Disc_img_LR.train()
    model_Disc_img_HR.train()
    
    running_loss_D_total = 0.0
    running_loss_G_total = 0.0

    running_loss_align = 0.0
    running_loss_rec = 0.0
    running_loss_res = 0.0
    running_loss_sty = 0.0
    running_loss_idt = 0.0
    running_loss_cyc = 0.0

    iter = 0    

    for data in tqdm(train_loader):
        iter += 1

        ########################
        #       data load      #
        ########################
        X_t, Y_s = data['img_LQ'], data['img_GT']

        ds4 = nn.Upsample(scale_factor=1/args.scale, mode='bicubic')
        X_s = ds4(Y_s)

        X_t = X_t.cuda(non_blocking=True)
        X_s = X_s.cuda(non_blocking=True)
        Y_s = Y_s.cuda(non_blocking=True)

        # real label and fake label
        batch_size = X_t.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=X_t.dtype).cuda(non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=X_t.dtype).cuda(non_blocking=True)


        ########################
        # (1) Update D network #
        ########################
        model_Disc_feat.zero_grad()
        model_Disc_img_LR.zero_grad()
        model_Disc_img_HR.zero_grad()

        for i in range(args.n_disc):
            # generator output (feature domain)
            F_t = model_Enc(X_t)
            F_s = model_Enc(X_s)

            # 1. feature aligment loss (discriminator)
            # output of discriminator (feature domain) (b x c(=1) x h x w)
            output_Disc_F_t = model_Disc_feat(F_t.detach())
            output_Disc_F_s = model_Disc_feat(F_s.detach())
            # discriminator loss (feature domain)
            loss_Disc_F_t = loss_MSE(output_Disc_F_t, fake_label)
            loss_Disc_F_s = loss_MSE(output_Disc_F_s, real_label)
            loss_Disc_feat_align = (loss_Disc_F_t + loss_Disc_F_s) / 2

            # 2. SR reconstruction loss (discriminator)
            # generator output (image domain)
            Y_s_s = model_Dec_SR(F_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_s_s = loss_MSE(output_Disc_Y_s_s, fake_label)
            loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_img_rec = (loss_Disc_Y_s_s + loss_Disc_Y_s) / 2

            # 4. Target degradation style loss
            # generator output (image domain)
            X_s_t = model_Dec_Id(F_s)
            # output of discriminator (image domain)
            output_Disc_X_s_t = model_Disc_img_LR(X_s_t.detach())
            output_Disc_X_t = model_Disc_img_LR(X_t)
            # discriminator loss (image domain)
            loss_Disc_X_s_t = loss_MSE(output_Disc_X_s_t, fake_label)
            loss_Disc_X_t = loss_MSE(output_Disc_X_t, real_label)
            loss_Disc_img_sty = (loss_Disc_X_s_t + loss_Disc_X_t) / 2

            # 6. Cycle loss
            # generator output (image domain)
            Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
            # output of discriminator (image domain)
            output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, fake_label)
            loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_img_cyc = (loss_Disc_Y_s_t_s + loss_Disc_Y_s) / 2

            # discriminator weight update
            loss_D_total = loss_Disc_feat_align + loss_Disc_img_rec + loss_Disc_img_sty + loss_Disc_img_cyc
            loss_D_total.backward()
            optimizer_D.step()



        scheduler_D.step()


        ########################
        # (2) Update G network #
        ########################
        model_Enc.zero_grad()
        model_Dec_Id.zero_grad()
        model_Dec_SR.zero_grad()

        for i in range(args.n_gen):
            # generator output (feature domain)
            F_t = model_Enc(X_t)
            F_s = model_Enc(X_s)

            # 1. feature alignment loss (generator)
            # output of discriminator (feature domain)
            output_Disc_F_t = model_Disc_feat(F_t)
            output_Disc_F_s = model_Disc_feat(F_s)
            # generator loss (feature domain)
            loss_G_F_t = loss_MSE(output_Disc_F_t, (real_label + fake_label)/2)
            loss_G_F_s = loss_MSE(output_Disc_F_s, (real_label + fake_label)/2)
            L_align_E = loss_G_F_t + loss_G_F_s

            # 2. SR reconstruction loss
            # generator output (image domain)
            Y_s_s = model_Dec_SR(F_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s)
            # L1 loss
            loss_L1_rec = loss_L1(Y_s.detach(), Y_s_s)
            # perceptual loss
            loss_percept_rec = loss_percept(Y_s.detach(), Y_s_s)
            # adversatial loss
            loss_G_Y_s_s = loss_MSE(output_Disc_Y_s_s, real_label)
            L_rec_G_SR = loss_L1_rec + args.lambda_percept*loss_percept_rec + args.lambda_adv*loss_G_Y_s_s

            # 3. Target LR restoration loss
            X_t_t = model_Dec_Id(F_t)
            L_res_G_t = loss_L1(X_t, X_t_t)

            # 4. Target degredation style loss
            # generator output (image domain)
            X_s_t = model_Dec_Id(F_s)
            # output of discriminator (img domain)
            output_Disc_X_s_t = model_Disc_img_LR(X_s_t)
            # generator loss (feature domain)
            loss_G_X_s_t = loss_MSE(output_Disc_X_s_t, real_label)
            L_sty_G_t = loss_G_X_s_t

            # 5. Feature identity loss
            F_s_tilda = model_Enc(model_Dec_Id(F_s))
            L_idt_G_t = loss_L1(F_s, F_s_tilda)

            # 6. Cycle loss
            # generator output (image domain)
            Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
            # output of discriminator (image domain)
            output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s)
            # L1 loss
            loss_L1_cyc = loss_L1(Y_s.detach(), Y_s_t_s)
            # perceptual loss
            loss_percept_cyc = loss_percept(Y_s.detach(), Y_s_t_s)
            # adversarial loss 
            loss_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, real_label)
            L_cyc_G_t_G_SR = loss_L1_cyc + args.lambda_percept*loss_percept_cyc + args.lambda_adv*loss_Y_s_t_s

            # generator weight update
            loss_G_total = args.lambda_align*L_align_E + args.lambda_rec*L_rec_G_SR + args.lambda_res*L_res_G_t + args.lambda_sty*L_sty_G_t + args.lambda_idt*L_idt_G_t + args.lambda_cyc*L_cyc_G_t_G_SR
            loss_G_total.backward()
            optimizer_G.step()
        scheduler_G.step()


        ########################
        #     compute loss     #
        ########################
        running_loss_D_total += loss_D_total.item()
        running_loss_G_total += loss_G_total.item()

        running_loss_align += L_align_E.item()
        running_loss_rec += L_rec_G_SR.item()
        running_loss_res += L_res_G_t.item()
        running_loss_sty += L_sty_G_t.item()
        running_loss_idt += L_idt_G_t.item()
        running_loss_cyc += L_cyc_G_t_G_SR.item()
        if iter % args.log_interval == 0:
            wandb.log(
                {
                    "loss_D_total_step": running_loss_D_total/iter,
                    "loss_G_total_step": running_loss_G_total/iter,
                    "loss_align_step": running_loss_align/iter,
                    "loss_rec_step": running_loss_rec/iter,
                    "loss_res_step": running_loss_res/iter,
                    "loss_sty_step": running_loss_sty/iter,
                    "loss_idt_step": running_loss_idt/iter,
                    "loss_cyc_step": running_loss_cyc/iter,
                }
            )
    ### EVALUATE ###
    total_PSNR = 0
    total_SSIM = 0
    total_LPIPS = 0
    val_iter = 0
    with torch.no_grad():
        model_Enc.eval()
        model_Dec_SR.eval()
        for batch_idx, batch in enumerate(val_loader):
            val_iter += 1
            source = batch["img_LQ"].to(device)
            target = batch["img_GT"].to(device)

            feat = model_Enc(source)
            out = model_Dec_SR(feat)

            total_PSNR += PSNR(out, target)
            total_SSIM += SSIM(out, target)
            total_LPIPS += LPIPS(out, target)
    
    wandb.log(
        {
            "epoch": epoch,
            "lr": optimizer_G.param_groups[0]['lr'],
            "loss_D_total_epoch": running_loss_D_total/iter,
            "loss_G_total_epoch": running_loss_G_total/iter,
            "loss_align_epoch": running_loss_align/iter,
            "loss_rec_epoch": running_loss_rec/iter,
            "loss_res_epoch": running_loss_res/iter,
            "loss_sty_epoch": running_loss_sty/iter,
            "loss_idt_epoch": running_loss_idt/iter,
            "loss_cyc_epoch": running_loss_cyc/iter,
            "PSNR_val": total_PSNR/val_iter,
            "SSIM_val": total_SSIM/val_iter,
            "LPIPS_val": total_LPIPS/val_iter
        }
    )


    if (epoch+1) % args.save_freq == 0:
        weights_file_name = 'epoch_%d.pth' % (epoch+1)
        weights_file = os.path.join(args.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,

            'model_Enc': model_Enc.state_dict(),
            'model_Dec_Id': model_Dec_Id.state_dict(),
            'model_Dec_SR': model_Dec_SR.state_dict(),
            'model_Disc_feat': model_Disc_feat.state_dict(),
            'model_Disc_img_LR': model_Disc_img_LR.state_dict(),
            'model_Disc_img_HR': model_Disc_img_HR.state_dict(),

            'optimizer_D': optimizer_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),

            'scheduler_D': scheduler_D.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))