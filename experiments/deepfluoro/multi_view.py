from pathlib import Path
import os

# import submitit
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
from multi_registration import MultiPoseRegressor
from diffdrr.pose import RigidTransform, se3_exp_map , se3_log_map


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

def compute_relative_transforms(offset1, offset2):
    # 计算从 offset1 到 offset2 的相对变换
    # 使用张量批处理操作计算所有相对变换
    relative_transforms = torch.matmul(torch.inverse(offset1), offset2)
    return relative_transforms



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

    model.train()
    draw_losses_list = []
    for epoch in range(n_epochs + 1):
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            contrast = contrast_distribution.sample().item()

            # 这里一次生成四个姿态用四个图片

            offset1 = get_random_offset(batch_size, device)
            pose_A = isocenter_pose.compose(offset1)
            # print ('\noffset1\n',offset1.matrix)
            # print('\npose_A\n',pose_A.matrix)


            offset2 = get_random_offset(batch_size, device)
            pose_B = isocenter_pose.compose(offset2)
            # print ('\noffset2\n',offset2.matrix)
            # print('\npose2\n',pose_B.matrix)
            # print('pose_first',pose.matrix[0])

            # # 将 offset1 和 offset2 合并成一个张量，形状为 (batch_size, 4, 4, 4)
            # offsets_combined = torch.stack([offset1.matrix, offset2.matrix], dim=0)

            # # 调整形状以便进行矩阵相乘，结果形状为 (batch_size * 2, 4, 4)
            # offsets_combined = offsets_combined.view(-1, 4, 4)

            # # 计算每组 SE 的相对变换
            # relative_transforms = compute_relative_transforms(offsets_combined[:, 0], offsets_combined[:, 1])

            # # 调整形状以便恢复每组的相对变换结果，形状为 (batch_size, 4, 4)
            # relative_transforms = relative_transforms.view(-1, 4, 4)
            
            # for i, relative_transform in enumerate(relative_transforms):
            #     print(f"相对变换 {i + 1}：\n{relative_transform}\n")

            log_poseA = se3_log_map(pose_A.matrix.transpose(1, 2))
            log_poseB = se3_log_map(pose_B.matrix.transpose(1, 2))

            log_delta = log_poseB - log_poseA

            # print('\ndelta\n',log_delta)
            # print('\njieguo\n',se3_exp_map(log_poseA + log_delta).transpose(1, 2), pose_B.matrix)

            # print('\n22222222222222222\n',torch.matmul(offset1.matrix,se3_exp_map(delta).transpose(1, 2)), pose_B.matrix)
            # print('\33333333333333333333\n',torch.matmul(offset1.matrix,se3_exp_map(delta).transpose(1, 2)), pose_B.matrix)

            img_A = drr(pose_A, bone_attenuation_multiplier=contrast)
            img_A = transforms(img_A)

            img_B = drr(pose_B, bone_attenuation_multiplier=contrast)
            img_B = transforms(img_B)

            pred_offset_A,pred_offset_B  = model(img_A, img_B)

            pred_pose_A = isocenter_pose.compose(pred_offset_A)
            pred_pose_B = isocenter_pose.compose(pred_offset_B)

            pred_img_A = drr(pred_pose_A)
            pred_img_B = drr(pred_pose_B)
            
            pred_img_A = transforms(pred_img_A)
            pred_img_B = transforms(pred_img_B)
            
            ncc_AA = metric(pred_img_A, img_A)
            ncc_BB = metric(pred_img_B, img_B)


            pred_img_A_via_B = drr(RigidTransform(se3_exp_map(se3_log_map(pred_pose_B.matrix.transpose(1, 2)) - log_delta).transpose(1, 2)))
            ncc_A_B = metric(pred_img_A_via_B, img_A)

            pred_img_B_via_A = drr(RigidTransform(se3_exp_map(se3_log_map(pred_pose_A.matrix.transpose(1, 2)) + log_delta).transpose(1, 2)))
            ncc_B_A = metric(pred_img_B_via_A, img_B)
            
            log_geodesic_AA = geodesic(pred_pose_A, pose_A)
            log_geodesic_BB = geodesic(pred_pose_B, pose_B)

            log_geodesic_A_via_B = geodesic(RigidTransform(se3_exp_map(se3_log_map(pred_pose_B.matrix.transpose(1, 2)) + log_delta).transpose(1, 2)), pose_A)
            log_geodesic_B_via_A = geodesic(RigidTransform(se3_exp_map(se3_log_map(pred_pose_A.matrix.transpose(1, 2)) - log_delta).transpose(1, 2)), pose_B)
            
            geodesic_rot_AA, geodesic_xyz_AA, double_geodesic_AA = double(pred_pose_A, pose_A)
            geodesic_rot_BB, geodesic_xyz_BB, double_geodesic_BB = double(pred_pose_B, pose_B)

            geodesic_rot_AB, geodesic_xyz_AB, double_geodesic_AB = double(RigidTransform(se3_exp_map(se3_log_map(pred_pose_B.matrix.transpose(1, 2)) + log_delta).transpose(1, 2)), pose_A)
            geodesic_rot_BA, geodesic_xyz_BA, double_geodesic_BA = double(RigidTransform(se3_exp_map(se3_log_map(pred_pose_A.matrix.transpose(1, 2)) - log_delta).transpose(1, 2)), pose_B)

            # loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
            loss = 0.25*((1 - ncc_AA)+(1 - ncc_BB)+(1 - ncc_A_B)+(1 - ncc_B_A)) + 0.25*1e-2 * (
                log_geodesic_AA + log_geodesic_BB+ log_geodesic_A_via_B+ log_geodesic_B_via_A+double_geodesic_AA + double_geodesic_BB+ double_geodesic_AB+ double_geodesic_BA)
            
            if loss.isnan().any():
                # print("Aaaaaaand we've crashed...")
                # print(ncc)
                # print(log_geodesic)
                # print(geodesic_rot)
                # print(geodesic_xyz)
                # print(double_geodesic)
                # print(pose.get_matrix())
                # print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "height": drr.detector.height,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose_A": pose_A.get_matrix().cpu(),
                        "pose_B": pose_B.get_matrix().cpu(),
                        "pred_pose_A": pred_pose_A.get_matrix().cpu(),
                        "pred_pose_B": pred_pose_B.get_matrix().cpu(),
                        "img_A": img_A.cpu(),
                        "img_B": img_B.cpu(),
                        "pred_img_A": pred_img_A.cpu(),
                        "pred_img_B": pred_img_B.cpu(),
                        **model_params,
                    },
                    f"checkpoints_multiview/specimen_{id_number:02d}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            # itr.set_postfix(
            #     geodesic_rot=geodesic_rot.mean().item(),
            #     geodesic_xyz=geodesic_xyz.mean().item(),
            #     geodesic_dou=double_geodesic.mean().item(),
            #     geodesic_se3=log_geodesic.mean().item(),
            #     loss=loss.mean().item(),
            #     ncc=ncc.mean().item(),
            # )

        losses = torch.tensor(losses)
        # tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")

        draw_losses = round(losses.mean().item(),4)
        draw_losses_list.append(draw_losses)


        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints_multiview/specimen_{id_number:02d}_best.ckpt",
            )

        if epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints_multiview/specimen_{id_number:02d}_epoch{epoch:03d}.ckpt",
            )

        # 在训练完成后，绘制 draw_losses 曲线
    plt.plot(range(1, n_epochs+2), draw_losses_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # 保存绘制的曲线并关闭绘图窗口
    plt.savefig(f"/home/data/cyx/autodl-tmp/DiffPose_copy/experiments/deepfluoro/loss_multiview/{id_number}.png")
    plt.close()

def main(
    id_number,
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=2,
    n_epochs=1000,
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
    model = MultiPoseRegressor(**model_params)
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
    # id_numbers = [1]
    # id_numbers = [1, 2, 3, 4, 5, 6]
    id_numbers = [2, 3, 4, 5, 6]
    
    # pretrained_checkpoint = "/home/data/cyx/autodl-tmp/DiffPose-main/experiments/deepfluoro/checkpoints_new/specimen_01_epoch800.ckpt"
    Path("checkpoints_multiview").mkdir(exist_ok=True)
    Path("loss_multiview").mkdir(exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 使用一个简单的循环代替 submitit 的 map_array
    for id_number in id_numbers:
        main(id_number)