from pathlib import Path
import os

# import submitit
import pandas as pd

import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Transforms, get_random_offset, Evaluator
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor

import matplotlib.pyplot as plt

# import sys
# sys.path.append('.../diffusion')
from diffusion import DiffusionScheduler as DS , AttrDict
from diffdrr.pose import RigidTransform, se3_exp_map , se3_log_map
# from ...diffusion.diffusion import DiffusionScheduler as DS , AttrDict

def init_opts(opts):
    
    opts.schedule_type = ["linear", "cosine"][1]

    opts.n_diff_steps = 200
    opts.beta_1 = 0.001
    opts.beta_T = 0.02

    opts.sigma_r = 0.1
    opts.sigma_t = 0.01
    opts.is_add_noise = True
    return opts

def load(id_number, height, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.focal_len / 2,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr

def evaluate(specimen, isocenter_pose, model, transforms, device):
    error = []
    model.eval()
    for idx in tqdm(range(len(specimen)), ncols=100):
        target_registration_error = Evaluator(specimen, idx)
        img, _ = specimen[idx]
        img = img.to(device)
        img = transforms(img)
        with torch.no_grad():
            offset = model(img)
        pred_pose = isocenter_pose.compose(offset)
        mtre = target_registration_error(pred_pose.cpu()).item()
        error.append(mtre)
    return error


def train(
    id_number,
    model,
    optimizer,
    scheduler,
    drr,
    transforms,
    specimen,
    isocenter_pose,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
):
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    best_loss = torch.inf

    opts = AttrDict()
    opts = init_opts(opts)
    opts.vs = DS(opts)


    model.train()
    draw_losses_list = []
    for epoch in range(n_epochs + 1):
        losses = []
        # 显示进度条
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            contrast = contrast_distribution.sample().item()

            # 这里一次生成四个姿态用四个图片
            offset = get_random_offset(batch_size, device)
            # print ('offset',offset.matrix)
            # print ('se3_log_map',se3_log_map(offset.matrix))
            # offset_list.append(offset.matrix)
            # print(epoch,'\n')
            # print(offset_list)


            pose = isocenter_pose.compose(offset)
            # print('pose',pose.matrix)
            # print('isocenter_pose',isocenter_pose.matrix)
            # print('pose_first',pose.matrix[0])


            B = batch_size
            H_0 = torch.eye(4)[None].expand(B, -1, -1).to(device)  # 初始姿态矩阵
            H_0[:, :3, :4] = pose.matrix[:, :3, :4]
            # print('H_0',H_0)

            H_T = torch.eye(4)[None].expand(B, -1, -1).to(device)
            H_T[:, :3, :4] = isocenter_pose.matrix[:, :3, :4]
            taus = opts.vs.uniform_sample_t(B)

            alpha_bars = opts.vs.alpha_bars[taus].to(device)[:, None]

            # print('H_T @ torch.inverse(H_0).transpose(1, 2)', (H_T @ torch.inverse(H_0)).transpose(1, 2))
            H_t = se3_exp_map((1. - torch.sqrt(alpha_bars)) * se3_log_map((H_T @ torch.inverse(H_0)).transpose(1, 2))).transpose(1, 2) @ H_0
            # print('H_t',H_t)

            scale = torch.cat([torch.ones(3) * opts.sigma_r, torch.ones(3) * opts.sigma_t])[None].to(device)
            noise = torch.sqrt(1. - alpha_bars) * scale * torch.randn(B, 6).to(device)
            
            H_noise = se3_exp_map(noise).transpose(1, 2)
            H_t_noise = RigidTransform((H_noise @ H_t).to(device))

            # print('H_t_noise', H_t_noise)

            img = drr(pose, bone_attenuation_multiplier=contrast)
            img = transforms(img)

            pred_offset = model(img)
            # print('pred_offset',pred_offset.matrix)
            pred_pose = isocenter_pose.compose(pred_offset)
            # print('pred_pose',pred_pose)
            pred_img = drr(pred_pose)
            
            ncc = metric(pred_img, img)

            img_noise = drr(H_t_noise)
            img_noise = transforms(img_noise)

            pred_offset_noise = model(img_noise)
            pred_pose_nosise = isocenter_pose.compose(pred_offset_noise)
            pred_img_noise = drr(pred_pose_nosise)
            pred_img_noise = transforms(pred_img_noise)
            ncc_noise = metric(pred_img_noise, img_noise)

            log_geodesic_noise = geodesic(pred_pose_nosise, pose)
            log_geodesic = geodesic(pred_pose, H_t_noise)

            loss = 2 - ncc - ncc_noise + 1e-1*(log_geodesic_noise+log_geodesic)



            # if loss.isnan().any():
            #     print("Aaaaaaand we've crashed...")
            #     print(ncc)
            #     # print(log_geodesic)
            #     # print(geodesic_rot)
            #     # print(geodesic_xyz)
            #     # print(double_geodesic)
            #     print(pose.get_matrix())
            #     print(pred_pose.get_matrix())
            #     torch.save(
            #         {
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "height": drr.detector.height,
            #             "epoch": epoch,
            #             "batch_size": batch_size,
            #             "n_epochs": n_epochs,
            #             "n_batches_per_epoch": n_batches_per_epoch,
            #             "pose": pose.get_matrix().cpu(),
            #             "pred_pose": pred_pose.get_matrix().cpu(),
            #             "img": img.cpu(),
            #             "pred_img": pred_img.cpu()
            #             **model_params,
            #         },
            #         f"checkpoints_diffusion_200/specimen_{id_number:02d}_crashed.ckpt",
            #     )
            #     raise RuntimeError("NaN loss")

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            # itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            # itr.set_postfix(
            #     # geodesic_rot=geodesic_rot.mean().item(),
            #     # geodesic_xyz=geodesic_xyz.mean().item(),
            #     # geodesic_dou=double_geodesic.mean().item(),
            #     # geodesic_se3=log_geodesic.mean().item(),
            #     loss=loss.mean().item(),
            #     ncc=ncc.mean().item(),
            # )

        losses = torch.tensor(losses)
        # tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")

        draw_losses = round(losses.mean().item(),4)
        draw_losses_list.append(draw_losses)


        # if losses.mean() < best_loss and not losses.isnan().any():
        #     best_loss = losses.mean().item()
        #     torch.save(
        #         {
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "height": drr.detector.height,
        #             "epoch": epoch,
        #             "loss": losses.mean().item(),
        #             "batch_size": batch_size,
        #             "n_epochs": n_epochs,
        #             "n_batches_per_epoch": n_batches_per_epoch,
        #             **model_params,
        #         },
        #         f"checkpoints_diffusion_200/specimen_{id_number:02d}_best.ckpt",
        #     )

        # if epoch % 50 == 0:
        #     torch.save(
        #         {
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "height": drr.detector.height,
        #             "epoch": epoch,
        #             "loss": losses.mean().item(),
        #             "batch_size": batch_size,
        #             "n_epochs": n_epochs,
        #             "n_batches_per_epoch": n_batches_per_epoch,
        #             **model_params,
        #         },
        #         f"checkpoints_diffusion_200/specimen_{id_number:02d}_epoch{epoch:03d}.ckpt",
        #     )
            



        # 在训练完成后，绘制 draw_losses 曲线
    # plt.plot(range(1, n_epochs+2), draw_losses_list, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.grid(True)

    # # 保存绘制的曲线并关闭绘图窗口
    # plt.savefig(f"/home/data/cyx/autodl-tmp/DiffPose_copy/experiments/deepfluoro/loss_diffusion_200/lr1e-1_{id_number}.png")
    # plt.close()


def main(
    id_number,
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=4,
    n_epochs=500,
    n_batches_per_epoch=100,
):
    id_number = int(id_number)

    device = torch.device("cuda")
    specimen, isocenter_pose, transforms, drr = load(id_number, height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )
    
    train(
        id_number,
        model,
        optimizer,
        scheduler,
        drr,
        transforms,
        specimen,
        isocenter_pose,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
    )


if __name__ == "__main__":
    id_numbers = [1]
    # id_numbers = [1, 2, 3, 4, 5, 6]
    
    # pretrained_checkpoint = "/home/data/cyx/autodl-tmp/DiffPose-main/experiments/deepfluoro/checkpoints_new/specimen_01_epoch800.ckpt"
    Path("checkpoints_diffusion_200").mkdir(exist_ok=True)
    Path("loss_diffusion_200").mkdir(exist_ok=True)
    Path("train_offset").mkdir(exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # 使用一个简单的循环代替 submitit 的 map_array
    for id_number in id_numbers:
        offset_list = []
        main(id_number)

        df = pd.DataFrame(offset_list)
        df.to_csv(f"train_offset/subject{id_number}.csv", index=False)



# if __name__ == "__main__":
#     id_numbers = [1, 2, 3, 4, 5, 6]
#     Path("checkpoints").mkdir(exist_ok=True)

#     executor = submitit.AutoExecutor(folder="logs")
#     executor.update_parameters(
#         name="deepfluoro",
#         gpus_per_node=1,
#         mem_gb=43.5,
#         slurm_array_parallelism=len(id_numbers),
#         slurm_partition="A6000",
#         slurm_exclude="sumac,fennel",
#         timeout_min=10_000,
#     )
#     jobs = executor.map_array(main, id_numbers)
