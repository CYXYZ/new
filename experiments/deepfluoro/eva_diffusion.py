from pathlib import Path
import os

import pandas as pd
# import submitit
import torch
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms, get_random_offset
from diffpose.registration import PoseRegressor

from diffusion import DiffusionScheduler as DS , AttrDict
from diffdrr.pose import RigidTransform, se3_exp_map , se3_log_map
from diffdrr.drr import DRR
from diffdrr.renderers import Siddon

def load_specimen(id_number, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)
    return specimen, isocenter_pose

def init_opts(opts):
    
    opts.schedule_type = ["linear", "cosine"][1]

    opts.n_diff_steps = 200
    opts.beta_1 = 0.001
    opts.beta_T = 0.02

    opts.sigma_r = 0.1
    opts.sigma_t = 0.01
    opts.is_add_noise = True
    return opts

def load_model(model_name, device):
    ckpt = torch.load(model_name)
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    transforms = Transforms(ckpt["height"])
    return model, transforms


def evaluate(specimen, isocenter_pose, model, transforms, device):
    error = []
    opts = AttrDict()
    opts = init_opts(opts)
    opts.vs = DS(opts)
    model.eval()
    height=256
    subsample = (1536 - 100) / 256
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

    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    

    # 这里一次生成四个姿态用四个图片
    




    for idx in tqdm(range(len(specimen)), ncols=100):
        print('\n',len(specimen),idx,'\n')
        target_registration_error = Evaluator(specimen, idx)


        contrast = contrast_distribution.sample().item()
        offset = get_random_offset(100, device)
        pose = isocenter_pose.compose(offset)
        # img, _ = specimen[idx]
        img = drr(pose, bone_attenuation_multiplier=contrast)
        img = img.to(device)
        img = transforms(img)

        # img = img.to(device)
        # img_t = transforms(img)
        # with torch.no_grad():
        #     offset = model(img_t)
        # pred_pose = isocenter_pose.compose(offset)

        img_list = []
        offset = get_random_offset(1,device)
        H_t = isocenter_pose.compose(offset).matrix.to(device)

        delta_H_0 = model(img)

        for t in range (5, 1, -1):
            with torch.no_grad():
                img_list.append(img_t.detach().cpu().numpy())
                img_t = drr(RigidTransform(H_t))
                
                delta_H_t = model(img_t)

                # offset = model(img_t)
                # _delta_H_t = RigidTransform(H_t).compose(offset).matrix.to(device)

                # H_0 = _delta_H_t @ H_t
                gamma0 = opts.vs.gamma0[t]
                gamma1 = opts.vs.gamma1[t]

                H_t = se3_exp_map(gamma0 * se3_log_map(isocenter_pose.compose(delta_H_0).matrix.to(device).transpose(1, 2)) + gamma1 * se3_log_map(isocenter_pose.compose(delta_H_t).matrix.to(device).transpose(1, 2))).transpose(1, 2)
                # print('H_txxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',H_t)
                # alpha_bar = opts.vs.alpha_bars[t]
                # alpha_bar_ = opts.vs.alpha_bars[t-1]
                # beta = opts.vs.betas[t]
                
                # cc = ((1 - alpha_bar_) / (1.- alpha_bar)) * beta
                # print('\n文章里面的lamda0',cc)

                # scale = torch.cat([torch.ones(3) * opts.sigma_r, torch.ones(3) * opts.sigma_t])[None].to(device)  
                # print('\nscale',scale)

                # noise = torch.sqrt(cc) * scale * torch.randn(1, 6).to(device)  # [B, 6]
                # H_noise = se3_exp_map(noise).transpose(1, 2)
                # print('\nH_noise',H_noise)
                # H_t = H_noise @ H_t  # [B, 4, 4]

                img_t = drr(RigidTransform(H_t)).to(device)

        print('\n', H_t)
        mtre = target_registration_error(RigidTransform(H_t).cpu()).item()

        # mtre = target_registration_error(pred_pose.cpu()).item()
        print("全部feducials的mtre",mtre)
        
        error.append(mtre)
    return error


def main(id_number):
    device = torch.device("cuda")
    specimen, isocenter_pose = load_specimen(id_number, device)
    models = sorted(Path("checkpoints_diffusion/").glob(f"specimen_{id_number:02d}_epoch*.ckpt"))

    errors = []
    for model_name in models:
        model, transforms = load_model(model_name, device)
        error = evaluate(specimen, isocenter_pose, model, transforms, device)
        errors.append([model_name.stem] + error)

    df = pd.DataFrame(errors)
    df.to_csv(f"eval_diffusion_200/subject{id_number}.csv", index=False)


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Path("eval_diffusion_200").mkdir(exist_ok=True)
    id_numbers = [1]
    # id_numbers = [1, 2, 3, 4, 5, 6]
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # 使用一个简单的循环代替 submitit 的 map_array
    for id_number in id_numbers:
        main(id_number)



    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(
    #     name="eval",
    #     gpus_per_node=1,
    #     mem_gb=10.0,
    #     slurm_array_parallelism=len(id_numbers),
    #     slurm_exclude="curcum",
    #     slurm_partition="2080ti",
    #     timeout_min=10_000,
    # )
    # jobs = executor.map_array(main, id_numbers)
