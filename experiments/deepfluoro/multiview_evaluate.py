from pathlib import Path
import os

import pandas as pd
# import submitit
import torch
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from multi_registration import MultiPoseRegressor


def load_specimen(id_number, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)
    return specimen, isocenter_pose


def load_model(model_name, device):
    ckpt = torch.load(model_name)
    model = MultiPoseRegressor(
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
    model.eval()
    for idx in tqdm(range(len(specimen)-1), ncols=100):
        print('\n',len(specimen),idx,'\n')
        target_registration_error_A = Evaluator(specimen, idx)
        target_registration_error_B = Evaluator(specimen, idx+1)
        
        img_A, _ = specimen[idx]
        img_A = img_A.to(device)
        img_A = transforms(img_A)

        img_B, _ = specimen[idx+1]
        img_B = img_B.to(device)
        img_B = transforms(img_B)
        
        with torch.no_grad():
            pred_offset_A,pred_offset_B  = model(img_A, img_B)

        pred_pose_A = isocenter_pose.compose(pred_offset_A)
        pred_pose_B = isocenter_pose.compose(pred_offset_B)

        mtre_A = target_registration_error_A(pred_pose_A.cpu()).item()
        mtre_B = target_registration_error_B(pred_pose_B.cpu()).item()

        print("全部feducials的mtre",mtre_A,mtre_B)
        error.append(mtre_A)
        error.append(mtre_B)
    return error


def main(id_number):
    device = torch.device("cuda")
    specimen, isocenter_pose = load_specimen(id_number, device)
    # models = sorted(Path("checkpoints/").glob(f"specimen_02_epoch*.ckpt"))
    models = sorted(Path("checkpoints_multiview/").glob(f"specimen_{id_number:02d}_epoch*.ckpt"))
    # models = sorted(Path("checkpoints_diffusion_1/").glob(f"specimen_{id_number:02d}_epoch*.ckpt"))

    errors = []
    for model_name in models:
        model, transforms = load_model(model_name, device)
        error = evaluate(specimen, isocenter_pose, model, transforms, device)
        errors.append([model_name.stem] + error)

    df = pd.DataFrame(errors)
    df.to_csv(f"multiview_evaluate/subject{id_number}.csv", index=False)


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Path("multiview_evaluate").mkdir(exist_ok=True)
    # id_numbers = [1, 2, 3, 4, 5, 6]
    id_numbers = [1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
