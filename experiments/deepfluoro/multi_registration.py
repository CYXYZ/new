import timm
import torch

# %% ../notebooks/api/03_registration.ipynb 5
from diffdrr.pose import RigidTransform, convert


class MultiPoseRegressor(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        output = self.backbone(torch.randn(1, 1, 256, 256)).shape[-1]
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)
        # self.xyz_regression_with_input = torch.nn.Linear(output, 4)
        # self.rot_regression_with_input = torch.nn.Linear(output, n_angular_components+1)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        rot1 = self.rot_regression(x1)
        xyz1 = self.xyz_regression(x1)

        x2 = self.backbone(x2)
        rot2 = self.rot_regression(x2)
        xyz2 = self.xyz_regression(x2)

        return convert(
            rot1,
            xyz1,
            parameterization=self.parameterization,
            convention=self.convention,
        ),convert(
            rot2,
            xyz2,
            parameterization=self.parameterization,
            convention=self.convention,
        )

# %% ../notebooks/api/03_registration.ipynb 6
N_ANGULAR_COMPONENTS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "se3_log_map": 3,
    "quaternion": 4,
    "rotation_6d": 6,
    "rotation_10d": 10,
    "quaternion_adjugate": 10,
}