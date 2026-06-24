import torch
from .registry import register

_TORCHLOGIX_IMPORT_ERROR = None
try:
    from torchlogix.layers import LogicConv2d, LogicDense, GroupSum, FixedBinarization, LearnableBinarization, OrPooling2d
except ImportError as exc:
    _TORCHLOGIX_IMPORT_ERROR = exc

    class _MissingTorchLogixLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("torchlogix is required for LGN models.") from _TORCHLOGIX_IMPORT_ERROR

    LogicConv2d = LogicDense = GroupSum = FixedBinarization = LearnableBinarization = OrPooling2d = _MissingTorchLogixLayer


# models like in paper: https://iopscience.iop.org/article/10.1088/2632-2153/ad6a00/pdf
# Some text extracts at the bottom of this file

@register("towards-model-1", input_type="y-size", task_type="classification")
class TowardsModel1(torch.nn.Module):
    def __init__(self):
        super(TowardsModel1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 2), f"Expected input shape (batch_size, 2), but got {x.shape}"
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return self.softmax(x)


@register("towards-model-2", input_type="y-profile", task_type="classification")
class TowardsModel2(torch.nn.Module):
    def __init__(self):
        super(TowardsModel2, self).__init__()
        self.fc1 = torch.nn.Linear(14, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 14), f"Expected input shape (batch_size, 14), but got {x.shape}"
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

@register("towards-model-3", input_type="y-profile-timing", task_type="classification")
class TowardsModel3(torch.nn.Module):
    def __init__(self):
        super(TowardsModel3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1)  
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1)  
        self.fc1 = torch.nn.Linear(64 * 9 * 4 + 1, 32)  
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(32, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"

        y0 = x[:, 0:1] 
        profile = x[:, 1:].view(-1, 1, 13, 8)  

        x = torch.relu(self.conv1(profile))
        x = torch.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)  
        x = torch.cat((y0, x), dim=1)  
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def _import_brevitas_nn():
    try:
        import brevitas.nn as qnn
    except ImportError as exc:
        raise ImportError(
            "Brevitas is required for quantized towards models. "
            "Install dependencies from requirements.txt before using them."
        ) from exc
    return qnn


class QuantizedTowardsModel2(torch.nn.Module):
    weight_bit_width = None
    act_bit_width = None

    def __init__(self):
        super(QuantizedTowardsModel2, self).__init__()
        qnn = _import_brevitas_nn()
        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit_width)
        self.fc1 = qnn.QuantLinear(14, 128, weight_bit_width=self.weight_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=self.act_bit_width)
        self.fc2 = qnn.QuantLinear(128, 3, weight_bit_width=self.weight_bit_width)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 14), f"Expected input shape (batch_size, 14), but got {x.shape}"
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


class QuantizedTowardsModel3(torch.nn.Module):
    weight_bit_width = None
    act_bit_width = None

    def __init__(self):
        super(QuantizedTowardsModel3, self).__init__()
        qnn = _import_brevitas_nn()
        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit_width)
        self.conv1 = qnn.QuantConv2d(1, 16, kernel_size=3, stride=1, weight_bit_width=self.weight_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=self.act_bit_width)
        self.conv2 = qnn.QuantConv2d(16, 64, kernel_size=3, stride=1, weight_bit_width=self.weight_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=self.act_bit_width)
        self.fc1 = qnn.QuantLinear(64 * 9 * 4 + 1, 32, weight_bit_width=self.weight_bit_width)
        self.relu3 = qnn.QuantReLU(bit_width=self.act_bit_width)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = qnn.QuantLinear(32, 3, weight_bit_width=self.weight_bit_width)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"

        x = self.quant_inp(x)
        y0 = x[:, 0:1]
        profile = x[:, 1:].view(-1, 1, 13, 8)

        x = self.relu1(self.conv1(profile))
        x = self.relu2(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


@register("towards-model-2-brevitas-w5a10", input_type="y-profile", task_type="classification")
class QuantizedTowardsModel2W5A10(QuantizedTowardsModel2):
    weight_bit_width = 5
    act_bit_width = 10


@register("towards-model-2-brevitas-w4a8", input_type="y-profile", task_type="classification")
class QuantizedTowardsModel2W4A8(QuantizedTowardsModel2):
    weight_bit_width = 4
    act_bit_width = 8


@register("towards-model-2-brevitas-w2a6", input_type="y-profile", task_type="classification")
class QuantizedTowardsModel2W2A6(QuantizedTowardsModel2):
    weight_bit_width = 2
    act_bit_width = 6


@register("towards-model-2-brevitas-w2a4", input_type="y-profile", task_type="classification")
class QuantizedTowardsModel2W2A4(QuantizedTowardsModel2):
    weight_bit_width = 2
    act_bit_width = 4


@register("towards-model-3-brevitas-w5a10", input_type="y-profile-timing", task_type="classification")
class QuantizedTowardsModel3W5A10(QuantizedTowardsModel3):
    weight_bit_width = 5
    act_bit_width = 10


@register("towards-model-3-brevitas-w4a8", input_type="y-profile-timing", task_type="classification")
class QuantizedTowardsModel3W4A8(QuantizedTowardsModel3):
    weight_bit_width = 4
    act_bit_width = 8


@register("towards-model-3-brevitas-w2a6", input_type="y-profile-timing", task_type="classification")
class QuantizedTowardsModel3W2A6(QuantizedTowardsModel3):
    weight_bit_width = 2
    act_bit_width = 6


@register("towards-model-3-brevitas-w2a4", input_type="y-profile-timing", task_type="classification")
class QuantizedTowardsModel3W2A4(QuantizedTowardsModel3):
    weight_bit_width = 2
    act_bit_width = 4


@register("lgn-dense-1", input_type="y-size", task_type="classification")
class DenseLGN1(torch.nn.Sequential):
    n_bits = 5
    k = 256  # scales the model size

    def __init__(self, thresholds):
        super(DenseLGN1, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1")
        param_kwargs = {
            "weight_init": "residual",
        }
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits*2, out_dim=3*self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3*self.k, out_dim=3*self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3*self.k, out_dim=3*self.k, parametrization_kwargs=param_kwargs),
        ]
        super(DenseLGN1, self).__init__(*layers, GroupSum(3, tau=1.0))


@register("lgn-dense-2", input_type="y-profile", task_type="classification")
class DenseLGN2(torch.nn.Sequential):
    n_bits = 5
    k = 1024

    def __init__(self, thresholds):
        super(DenseLGN2, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2")
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits*14, out_dim=3*self.k),
            LogicDense(in_dim=3*self.k, out_dim=3*self.k),
            LogicDense(in_dim=3*self.k, out_dim=3*self.k),
        ]
        super(DenseLGN2, self).__init__(*layers, GroupSum(3, tau=10.0))



@register("lgn-conv-3", input_type="y-profile-timing", task_type="classification")
class ConvLGN3(torch.nn.Module):
    n_bits = 5

    def __init__(self, thresholds):
        device = "cuda"
        super(ConvLGN3, self).__init__()
        self.n_bits = 5
        self.binarization = FixedBinarization(thresholds=thresholds, device=device)
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)
        self.fc1 = LogicDense(in_dim=128 * 9 * 4 + 5, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.group_sum = GroupSum(3, tau=10.0, device=device)


    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
    
        # print(f"Input shape: {x.shape}")
        x = self.binarization(x)
        # print(f"after binarization: {x.shape}")

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        # print(f"shape before conv1: {profile.shape}")

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # print(f"before group sum: {x.shape}")
        # print(x)

        return self.group_sum(x)
    



############### CNN ALL TIME SLICES 


@register("towards-model-3-20", input_type="y-profile-timing-20", task_type="classification")
class TowardsModel3_20(torch.nn.Module):
    def __init__(self):
        super(TowardsModel3_20, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.15)

        # Input profile shape: (1, 13, 20)
        # After conv1:            (32, 13, 20)
        # After pool:             (32, 6, 10)
        # After conv2:            (64, 6, 10)
        # After conv3:            (128, 6, 10)
        # After second pool:      (128, 3, 5)
        self.fc1 = torch.nn.Linear(128 * 3 * 5 + 1, 256)
        self.fc2 = torch.nn.Linear(256, 96)
        self.fc3 = torch.nn.Linear(96, 3)

    def forward(self, x):
        assert x.shape == (x.shape[0], 261), f"Expected input shape (batch_size, 261), but got {x.shape}"

        y0 = x[:, 0:1]
        profile = x[:, 1:].view(-1, 1, 13, 20)

        x = torch.relu(self.conv1(profile))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    



@register("towards-model-3-20-raw", input_type="y-profile-timing-20-raw", task_type="classification")
class TowardsModel3_20Raw(torch.nn.Module):
    def __init__(self):
        super(TowardsModel3_20Raw, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.15)

        # Input profile shape: (1, 13, 20)
        # After conv1:            (32, 13, 20)
        # After pool:             (32, 6, 10)
        # After conv2:            (64, 6, 10)
        # After conv3:            (128, 6, 10)
        # After second pool:      (128, 3, 5)
        self.fc1 = torch.nn.Linear(128 * 3 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 96)
        self.fc3 = torch.nn.Linear(96, 3)

    def forward(self, x):
        assert x.shape == (x.shape[0], 260), f"Expected input shape (batch_size, 260), but got {x.shape}"

        profile = x.view(-1, 1, 13, 20)

        x = torch.relu(self.conv1(profile))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


################################ All 20 Slices test (different tau + one contains y0 one doesnt) Bigger models

class _ConvLGNDeep100_20BigBase(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds, tau=20.0, use_y0=True):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.use_y0 = use_y0
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        # deeper + significantly wider
        k1, k2, k3, k4, k5, k6 = 384, 768, 1024, 1536, 1536, 1024

        self.conv1 = LogicConv2d(
            in_dim=(13, 20), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 20), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 10), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 10), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool2 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = LogicConv2d(
            in_dim=(3, 5), channels=k4, num_kernels=k5,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv6 = LogicConv2d(
            in_dim=(3, 5), channels=k5, num_kernels=k6,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        conv_flat = k6 * 3 * 5
        fc_in = self.n_bits + conv_flat if self.use_y0 else conv_flat

        # all divisible by 3
        self.fc1 = LogicDense(
            in_dim=fc_in, out_dim=18000,
            device=device, parametrization_kwargs=pk, lut_rank=2
        )
        self.fc2 = LogicDense(
            in_dim=18000, out_dim=18000,
            device=device, parametrization_kwargs=pk, lut_rank=2
        )
        self.fc3 = LogicDense(
            in_dim=18000, out_dim=9000,
            device=device, parametrization_kwargs=pk, lut_rank=2
        )
        self.fc4 = LogicDense(
            in_dim=9000, out_dim=4500,
            device=device, parametrization_kwargs=pk, lut_rank=2
        )
        self.fc5 = LogicDense(
            in_dim=4500, out_dim=3000,
            device=device, parametrization_kwargs=pk, lut_rank=2
        )

        self.group_sum = GroupSum(3, tau=tau, device=device)

    def forward(self, x):
        x = self.bin(x)

        if self.use_y0:
            y0 = x[:, 0:self.n_bits]
            p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 20)
        else:
            p = x.view(-1, self.n_bits, 13, 20)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)

        z = self.conv3(z)
        z = self.conv4(z)
        z = self.or_pool2(z)

        z = self.conv5(z)
        z = self.conv6(z)

        z = z.view(z.size(0), -1)

        if self.use_y0:
            z = torch.cat((y0, z), dim=1)

        z = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z)))))
        return self.group_sum(z)
    

@register("lgn-conv-6-100-orpool_tau10", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20Big_tau10(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=10.0, use_y0=True)


@register("lgn-conv-6-100-orpool_tau20", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20Big_tau20(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=20.0, use_y0=True)


@register("lgn-conv-6-100-orpool_tau40", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20Big_tau40(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=40.0, use_y0=True)


@register("lgn-conv-6-100-orpool_raw_tau10", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20Big_raw_tau10(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=10.0, use_y0=False)


@register("lgn-conv-6-100-orpool_raw_tau20", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20Big_raw_tau20(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=20.0, use_y0=False)


@register("lgn-conv-6-100-orpool_raw_tau40", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20Big_raw_tau40(_ConvLGNDeep100_20BigBase):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=40.0, use_y0=False)








################################ All 20 Slices test (different tau + one contains y0 one doesnt)




def _make_group_sum_tau(tau):
    return GroupSum(3, tau=float(tau), device="cuda")


class _ConvLGNDeep100_20Base(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds, tau=20.0, use_y0=True):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.use_y0 = use_y0
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960

        self.conv1 = LogicConv2d(
            in_dim=(13, 20), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 20), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 10), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 10), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        conv_flat = k4 * 3 * 5
        fc_in = self.n_bits + conv_flat if self.use_y0 else conv_flat

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=tau, device=device)

    def forward(self, x):
        x = self.bin(x)

        if self.use_y0:
            y0 = x[:, 0:self.n_bits]
            p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 20)
        else:
            p = x.view(-1, self.n_bits, 13, 20)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)

        if self.use_y0:
            z = torch.cat((y0, z), dim=1)

        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)


@register("lgn-conv-3-100-orpool_tau10", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20_tau10(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=10.0, use_y0=True)


@register("lgn-conv-3-100-orpool_tau20", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20_tau20(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=20.0, use_y0=True)


@register("lgn-conv-3-100-orpool_tau40", input_type="y-profile-timing-20", task_type="classification")
class ConvLGNDeep100_20_tau40(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=40.0, use_y0=True)


@register("lgn-conv-3-100-orpool_raw_tau10", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20_raw_tau10(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=10.0, use_y0=False)


@register("lgn-conv-3-100-orpool_raw_tau20", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20_raw_tau20(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=20.0, use_y0=False)


@register("lgn-conv-3-100-orpool_raw_tau40", input_type="y-profile-timing-20-raw", task_type="classification")
class ConvLGNDeep100_20_raw_tau40(_ConvLGNDeep100_20Base):
    n_bits = 100
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=40.0, use_y0=False)






















######################################## BATCH 128 + TAU TESTS 


@register("lgn-conv-3-100-orpool_128_2", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_128_2(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=2.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    



@register("lgn-conv-3-100-orpool_128_5", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_128_5(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=5.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    

@register("lgn-conv-3-100-orpool_128_10", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_128_10(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    

@register("lgn-conv-3-100-orpool_128_20", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_128_20(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=20.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    

@register("lgn-conv-3-100-orpool_128_40", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_128_40(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=40.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    




@register("lgn-dense-2-dense-100_128_-lessFeat_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_128__20(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=20.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)
    
@register("lgn-dense-2-dense-100_128_-lessFeat_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_128__40(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  

        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=40.0, device=device)

    def forward(self, x):
        x = self.bin(x)  
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)
    





@register("lgn-dense-2-dense-100_128_-lessFeat_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_128__10(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)


class _DenseOnlyLGN14F_Wide_128(torch.nn.Module):
    n_bits = 100
    k = 2000
    n_inputs = 14

    def __init__(self, thresholds, tau):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = self.n_inputs * self.n_bits

        h1 = 6 * self.k
        h2 = 6 * self.k
        h3 = 6 * self.k
        h4 = 4 * self.k
        h5 = 4 * self.k
        h6 = 4 * self.k
        out_dim = 3 * self.k

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=h1, out_dim=h2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=h2, out_dim=h3, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=h3, out_dim=h4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc5 = LogicDense(in_dim=h4, out_dim=h5, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc6 = LogicDense(in_dim=h5, out_dim=h6, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc7 = LogicDense(in_dim=h6, out_dim=out_dim, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=tau, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], self.n_inputs), f"Expected input shape (batch_size, {self.n_inputs}), but got {x.shape}"
        x = self.bin(x)
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Wide_128__5(_DenseOnlyLGN14F_Wide_128):
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=5.0)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Wide_128__10(_DenseOnlyLGN14F_Wide_128):
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=10.0)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Wide_128__20(_DenseOnlyLGN14F_Wide_128):
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=20.0)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_30", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Wide_128__30(_DenseOnlyLGN14F_Wide_128):
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=30.0)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Wide_128__40(_DenseOnlyLGN14F_Wide_128):
    def __init__(self, thresholds):
        super().__init__(thresholds=thresholds, tau=40.0)


class _DenseOnlyLGN14F_Model2FullSized(torch.nn.Module):
    n_bits = 100
    n_inputs = 14
    tau = 20.0
    hidden_dims = (3072, 3072, 3072)

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        if self.hidden_dims[-1] % 3 != 0:
            raise ValueError("The final LGN layer width must be divisible by 3.")

        self.bin = FixedBinarization(thresholds=thresholds)
        self.layers = torch.nn.ModuleList()

        prev_dim = self.n_inputs * self.n_bits
        for hidden_dim in self.hidden_dims:
            self.layers.append(
                LogicDense(
                    in_dim=prev_dim,
                    out_dim=hidden_dim,
                    device=device,
                    parametrization_kwargs=pk,
                    lut_rank=2,
                )
            )
            prev_dim = hidden_dim

        self.group_sum = GroupSum(3, tau=float(self.tau), device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], self.n_inputs), f"Expected input shape (batch_size, {self.n_inputs}), but got {x.shape}"
        z = self.bin(x)
        for layer in self.layers:
            z = layer(z)
        return self.group_sum(z)


class _DenseOnlyLGN14F_Model2FullS01DSNewLike(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (3072, 3072, 3072)


class _DenseOnlyLGN14F_Model2FullS02P163M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (6300, 6300, 6300, 4200, 4200, 4200, 3150)


class _DenseOnlyLGN14F_Model2FullS03P298M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (8600, 8600, 8600, 5700, 5700, 5700, 4302)


class _DenseOnlyLGN14F_Model2FullS04P434M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (10400, 10400, 10400, 6900, 6900, 6900, 5202)


class _DenseOnlyLGN14F_Model2FullS06P760M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (13800, 13800, 13800, 9200, 9200, 9200, 6900)


class _DenseOnlyLGN14F_Model2FullS07P968M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (15600, 15600, 15600, 10400, 10400, 10400, 7800)


class _DenseOnlyLGN14F_Model2FullS08P1202M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (17400, 17400, 17400, 11600, 11600, 11600, 8700)


class _DenseOnlyLGN14F_Model2FullS09P1460M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (19200, 19200, 19200, 12800, 12800, 12800, 9600)


class _DenseOnlyLGN14F_Model2FullS10P1744M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (21000, 21000, 21000, 14000, 14000, 14000, 10500)


class _DenseOnlyLGN14F_Model2FullS11P2054M(_DenseOnlyLGN14F_Model2FullSized):
    hidden_dims = (22800, 22800, 22800, 15200, 15200, 15200, 11400)


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS01DSNewLike__5(_DenseOnlyLGN14F_Model2FullS01DSNewLike):
    tau = 5.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS01DSNewLike__10(_DenseOnlyLGN14F_Model2FullS01DSNewLike):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS01DSNewLike__20(_DenseOnlyLGN14F_Model2FullS01DSNewLike):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike_30", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS01DSNewLike__30(_DenseOnlyLGN14F_Model2FullS01DSNewLike):
    tau = 30.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS01DSNewLike__40(_DenseOnlyLGN14F_Model2FullS01DSNewLike):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS02P163M__5(_DenseOnlyLGN14F_Model2FullS02P163M):
    tau = 5.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS02P163M__10(_DenseOnlyLGN14F_Model2FullS02P163M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS02P163M__20(_DenseOnlyLGN14F_Model2FullS02P163M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_30", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS02P163M__30(_DenseOnlyLGN14F_Model2FullS02P163M):
    tau = 30.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS02P163M__40(_DenseOnlyLGN14F_Model2FullS02P163M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS03P298M__5(_DenseOnlyLGN14F_Model2FullS03P298M):
    tau = 5.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS03P298M__10(_DenseOnlyLGN14F_Model2FullS03P298M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS03P298M__20(_DenseOnlyLGN14F_Model2FullS03P298M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M_30", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS03P298M__30(_DenseOnlyLGN14F_Model2FullS03P298M):
    tau = 30.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS03P298M__40(_DenseOnlyLGN14F_Model2FullS03P298M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS04P434M__5(_DenseOnlyLGN14F_Model2FullS04P434M):
    tau = 5.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS04P434M__10(_DenseOnlyLGN14F_Model2FullS04P434M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS04P434M__20(_DenseOnlyLGN14F_Model2FullS04P434M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M_30", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS04P434M__30(_DenseOnlyLGN14F_Model2FullS04P434M):
    tau = 30.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS04P434M__40(_DenseOnlyLGN14F_Model2FullS04P434M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS06P760M__10(_DenseOnlyLGN14F_Model2FullS06P760M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS06P760M__20(_DenseOnlyLGN14F_Model2FullS06P760M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS06P760M__40(_DenseOnlyLGN14F_Model2FullS06P760M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS06P760M__60(_DenseOnlyLGN14F_Model2FullS06P760M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS06P760M__80(_DenseOnlyLGN14F_Model2FullS06P760M):
    tau = 80.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS07P968M__10(_DenseOnlyLGN14F_Model2FullS07P968M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS07P968M__20(_DenseOnlyLGN14F_Model2FullS07P968M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS07P968M__40(_DenseOnlyLGN14F_Model2FullS07P968M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS07P968M__60(_DenseOnlyLGN14F_Model2FullS07P968M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS07P968M__80(_DenseOnlyLGN14F_Model2FullS07P968M):
    tau = 80.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS08P1202M__10(_DenseOnlyLGN14F_Model2FullS08P1202M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS08P1202M__20(_DenseOnlyLGN14F_Model2FullS08P1202M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS08P1202M__40(_DenseOnlyLGN14F_Model2FullS08P1202M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS08P1202M__60(_DenseOnlyLGN14F_Model2FullS08P1202M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS08P1202M__80(_DenseOnlyLGN14F_Model2FullS08P1202M):
    tau = 80.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS09P1460M__10(_DenseOnlyLGN14F_Model2FullS09P1460M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS09P1460M__20(_DenseOnlyLGN14F_Model2FullS09P1460M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS09P1460M__40(_DenseOnlyLGN14F_Model2FullS09P1460M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS09P1460M__60(_DenseOnlyLGN14F_Model2FullS09P1460M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS09P1460M__80(_DenseOnlyLGN14F_Model2FullS09P1460M):
    tau = 80.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS10P1744M__10(_DenseOnlyLGN14F_Model2FullS10P1744M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS10P1744M__20(_DenseOnlyLGN14F_Model2FullS10P1744M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS10P1744M__40(_DenseOnlyLGN14F_Model2FullS10P1744M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS10P1744M__60(_DenseOnlyLGN14F_Model2FullS10P1744M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS10P1744M__80(_DenseOnlyLGN14F_Model2FullS10P1744M):
    tau = 80.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M_10", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS11P2054M__10(_DenseOnlyLGN14F_Model2FullS11P2054M):
    tau = 10.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS11P2054M__20(_DenseOnlyLGN14F_Model2FullS11P2054M):
    tau = 20.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M_40", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS11P2054M__40(_DenseOnlyLGN14F_Model2FullS11P2054M):
    tau = 40.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M_60", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS11P2054M__60(_DenseOnlyLGN14F_Model2FullS11P2054M):
    tau = 60.0


@register("lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M_80", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_14F_Model2FullS11P2054M__80(_DenseOnlyLGN14F_Model2FullS11P2054M):
    tau = 80.0
















    



###############################TAU TESTS


@register("lgn-conv-3-100-orpool_2", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_2(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=2.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    

@register("lgn-conv-3-100-orpool_5", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_5(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=5.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)
    


@register("lgn-conv-3-100-orpool_20", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2_20(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=20.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)

@register("lgn-dense-2-dense-100-lessFeat_2", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_2(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=2.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)
    
@register("lgn-dense-2-dense-100-lessFeat_5", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_5(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=5.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)
    

@register("lgn-dense-2-dense-100-lessFeat_20", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide_20(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=20.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)

######################## new 3s

@register("lgn-conv-3-100-orpool", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 100 + 960*6 = 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)


@register("lgn-conv-3-100-orpool_cg", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep100_2cg(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 100
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        td = 3

        cg = {"channel_group_size": 4}

        k1, k2, k3, k4 = 256, 512, 1024, 960  # <-- k4 reduced to satisfy lut_rank=2

        self.conv1 = LogicConv2d(
            in_dim=(13, 8), channels=self.n_bits, num_kernels=k1,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk, connections_kwargs=cg,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8), channels=k1, num_kernels=k2,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk, connections_kwargs=cg,
        )

        self.or_pool1 = OrPooling2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = LogicConv2d(
            in_dim=(6, 4), channels=k2, num_kernels=k3,
            receptive_field_size=3, tree_depth=td, stride=1, padding=1,
            device=device, parametrization_kwargs=pk, connections_kwargs=cg,
        )
        self.conv4 = LogicConv2d(
            in_dim=(6, 4), channels=k3, num_kernels=k4,
            receptive_field_size=3, tree_depth=td, stride=2, padding=1,
            device=device, parametrization_kwargs=pk, connections_kwargs=cg,
        )

        fc_in = self.n_bits + (k4 * 3 * 2)  # 5860

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc2 = LogicDense(in_dim=3000*4, out_dim=3000*4, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc3 = LogicDense(in_dim=3000*4, out_dim=3000*2, device=device, parametrization_kwargs=pk, lut_rank=2)
        self.fc4 = LogicDense(in_dim=3000*2, out_dim=3000, device=device, parametrization_kwargs=pk, lut_rank=2)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv1(p)
        z = self.conv2(z)
        z = self.or_pool1(z)
        z = self.conv3(z)
        z = self.conv4(z)

        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)



####################### Model 2 with less input CONV + DENSE #######################
@register("lgn-dense-2-conv-100-lessFeat", input_type="y-profile", task_type="classification")
class DenseLGN2_100_DeepConv2_DeepFC5_k1000(torch.nn.Module):
    n_bits = 100
    k = 1500

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        # CHANGE: 14 -> 9
        self.conv1 = LogicConv2d(
            in_dim=(9, 1),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=(3, 1),
            tree_depth=3,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        # CHANGE: 12 -> 7 (because 9 -> 7 after conv1)
        self.conv2 = LogicConv2d(
            in_dim=(7, 1),
            channels=64,
            num_kernels=128,
            receptive_field_size=(3, 1),
            tree_depth=3,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        # CHANGE: 128*10*1 -> 128*5*1 (because 9 -> 7 -> 5)
        conv_out = 128 * 5 * 1

        self.fc1 = LogicDense(in_dim=conv_out, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=6 * self.k, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        b = x.shape[0]

        # CHANGE: 14 -> 9
        p = x.view(b, self.n_bits, 9, 1)

        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z)))))
        return self.group_sum(z)

@register("lgn-dense-2-dense-100-lessFeat", input_type="y-profile", task_type="classification")
class DenseOnlyLGN_9F_Wide(torch.nn.Module):
    n_bits = 100
    k = 2000  

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        in_dim = 9 * self.n_bits  # 450


        h1 = 6 * self.k   
        h2 = 6 * self.k   
        h3 = 6 * self.k   
        h4 = 4 * self.k   
        h5 = 4 * self.k   
        h6 = 4 * self.k  
        out_dim = 3 * self.k  

        self.fc1 = LogicDense(in_dim=in_dim, out_dim=h1, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=h1,    out_dim=h2, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=h2,    out_dim=h3, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=h3,    out_dim=h4, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=h4,    out_dim=h5, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=h5,    out_dim=h6, device=device, parametrization_kwargs=pk)
        self.fc7 = LogicDense(in_dim=h6,    out_dim=out_dim, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        # x: [B, 9]
        x = self.bin(x)  # expected: [B, 9*n_bits] = [B, 450]
        z = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))))
        return self.group_sum(z)


### MODELS REDUCED BIT FOR 3, INCREASED SIZE FOR 2 ###



@register("lgn-conv-3-10-deepconv4-deepfc6_tau10_fb_vwidth", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep10_23_vwidth(torch.nn.Module):
    n_bits = 10

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 10
        self.bin = FixedBinarization(thresholds=thresholds)

        pk = {"weight_init": "residual"}

        self.conv1 = LogicConv2d(
            in_dim=(13, 8),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8),
            channels=64,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        self.conv3 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=2,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (128 * 7 * 4)

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=9000, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=9000, out_dim=9000, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=9000, out_dim=6000, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=6000, out_dim=6000, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=6000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)

        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv4(self.conv3(self.conv2(self.conv1(p))))
        z = z.view(z.size(0), -1)              
        z = torch.cat((y0, z), dim=1)          
        z = self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z))))))

        return self.group_sum(z)

@register("lgn-conv-3-10-deepconv4-deepfc6_tau10_fb_swidth", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep10_23_swidth(torch.nn.Module):
    n_bits = 10

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 10
        self.bin = FixedBinarization(thresholds=thresholds)

        pk = {"weight_init": "residual"}

        self.conv1 = LogicConv2d(
            in_dim=(13, 8),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8),
            channels=64,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        self.conv3 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=4,
            stride=2,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (128 * 7 * 4)

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)

        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv4(self.conv3(self.conv2(self.conv1(p))))
        z = z.view(z.size(0), -1)              
        z = torch.cat((y0, z), dim=1)          
        z = self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z))))))

        return self.group_sum(z)




# @register("lgn-dense-2-50-deepconv2-deepfc5_k1000_tau10", input_type="y-profile", task_type="classification")
# class DenseLGN2_100_DeepConv2_DeepFC5_k1000(torch.nn.Module):
#     n_bits = 50
#     k = 1000             

#     def __init__(self, thresholds):
#         super().__init__()
#         print(f"Using {self.n_bits} bits for binarization in DenseLGN2_100_DeepConv2_DeepFC5_k512")
#         device = "cuda"

#         pk = {"weight_init": "residual"}

#         self.bin = FixedBinarization(thresholds=thresholds)

#         self.conv1 = LogicConv2d(
#             in_dim=(14, 1),
#             channels=self.n_bits,
#             num_kernels=64,
#             receptive_field_size=(3, 1),
#             tree_depth=2,
#             stride=1,
#             padding=0,
#             device=device,
#             parametrization_kwargs=pk,
#         )

#         self.conv2 = LogicConv2d(
#             in_dim=(12, 1),
#             channels=64,
#             num_kernels=128,
#             receptive_field_size=(3, 1),
#             tree_depth=2,
#             stride=1,
#             padding=0,
#             device=device,
#             parametrization_kwargs=pk,
#         )

#         conv_out = 128 * 10 * 1  

#         self.fc1 = LogicDense(in_dim=conv_out, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc2 = LogicDense(in_dim=6 * self.k, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc3 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc5 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)

#         self.group_sum = GroupSum(3, tau=10.0, device=device)

#     def forward(self, x):
#         x = self.bin(x)  

#         b = x.shape[0]
#         p = x.view(b, self.n_bits, 14, 1)

#         z = self.conv2(self.conv1(p))
#         z = z.view(z.size(0), -1)

#         z = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z)))))
#         return self.group_sum(z)



# @register("lgn-dense-2-50-deepconv2-densebetween-deepfc5_k1000_tau10", input_type="y-profile", task_type="classification")
# class DenseLGN2_50_DeepConv2_DenseBetween_DeepFC5_k1000(torch.nn.Module):
#     n_bits = 50
#     k = 1000

#     def __init__(self, thresholds):
#         super().__init__()
#         device = "cuda"
#         pk = {"weight_init": "residual"}

#         self.bin = FixedBinarization(thresholds=thresholds)

#         self.conv1 = LogicConv2d(
#             in_dim=(14, 1),
#             channels=self.n_bits,
#             num_kernels=64,
#             receptive_field_size=(3, 1),
#             tree_depth=2,
#             stride=1,
#             padding=0,
#             device=device,
#             parametrization_kwargs=pk,
#         )

#         # conv1 output: (b, 64, 12, 1) => 64*12*1 = 768
#         self.mid_fc = LogicDense(in_dim=64 * 12 * 1, out_dim=64 * 12 * 1, device=device, parametrization_kwargs=pk)

#         self.conv2 = LogicConv2d(
#             in_dim=(12, 1),
#             channels=64,
#             num_kernels=128,
#             receptive_field_size=(3, 1),
#             tree_depth=2,
#             stride=1,
#             padding=0,
#             device=device,
#             parametrization_kwargs=pk,
#         )

#         # conv2 output: (b, 128, 10, 1) => 128*10*1 = 1280
#         conv_out = 128 * 10 * 1

#         self.fc1 = LogicDense(in_dim=conv_out, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc2 = LogicDense(in_dim=6 * self.k, out_dim=6 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc3 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
#         self.fc5 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)

#         self.group_sum = GroupSum(3, tau=10.0, device=device)

#     def forward(self, x):
#         x = self.bin(x)

#         b = x.shape[0]
#         p = x.view(b, self.n_bits, 14, 1)

#         z = self.conv1(p)                 # (b, 64, 12, 1)
#         z = z.view(b, -1)                 # (b, 768)
#         z = self.mid_fc(z)                # (b, 768)
#         z = z.view(b, 64, 12, 1)          # back to conv2 input

#         z = self.conv2(z)                 # (b, 128, 10, 1)
#         z = z.view(b, -1)                 # (b, 1280)

#         z = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z)))))
#         return self.group_sum(z)
















################### New Trials ####################### 23/2/2025
@register("lgn-conv-3-200-deepconv4-deepfc4_tau10_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep200_23(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}

        self.conv1 = LogicConv2d(
            in_dim=(13, 8),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=3,
            tree_depth=3,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8),
            channels=64,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=3,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv3 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=3,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        self.conv4 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=3,
            stride=2,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (128 * 7 * 4)  

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv4(self.conv3(self.conv2(self.conv1(p))))
        z = z.view(z.size(0), -1)     
        z = torch.cat((y0, z), dim=1) 

        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)


@register("lgn-conv-3-50-deepconv4-deepfc6_tau10_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGNDeep50_23(torch.nn.Module):
    n_bits = 50

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 50
        self.bin = FixedBinarization(thresholds=thresholds)

        pk = {"weight_init": "residual"}

        self.conv1 = LogicConv2d(
            in_dim=(13, 8),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=3,
            tree_depth=2,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv2 = LogicConv2d(
            in_dim=(13, 8),
            channels=64,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=2,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        self.conv3 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=2,
            stride=1,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )
        self.conv4 = LogicConv2d(
            in_dim=(13, 8),
            channels=128,
            num_kernels=128,
            receptive_field_size=3,
            tree_depth=2,
            stride=2,
            padding=1,
            device=device,
            parametrization_kwargs=pk,
        )

        fc_in = self.n_bits + (128 * 7 * 4)

        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc6 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)

        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        z = self.conv4(self.conv3(self.conv2(self.conv1(p))))
        z = z.view(z.size(0), -1)              
        z = torch.cat((y0, z), dim=1)          
        z = self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z))))))

        return self.group_sum(z)



@register("lgn-dense-2-100-deepconv2-deepfc5_k512_tau8", input_type="y-profile", task_type="classification")
class DenseLGN2_100_DeepConv2_DeepFC5_k512(torch.nn.Module):
    n_bits = 100
    k = 512  # reduced width

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2_100_DeepConv2_DeepFC5_k512")
        device = "cuda"

        pk = {"weight_init": "residual"}

        self.bin = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(
            in_dim=(14, 1),
            channels=self.n_bits,
            num_kernels=64,
            receptive_field_size=(3, 1),
            tree_depth=2,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        self.conv2 = LogicConv2d(
            in_dim=(12, 1),
            channels=64,
            num_kernels=128,
            receptive_field_size=(3, 1),
            tree_depth=2,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        conv_out = 128 * 10 * 1  

        self.fc1 = LogicDense(in_dim=conv_out, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)
        self.fc5 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=8.0, device=device)

    def forward(self, x):
        x = self.bin(x)  

        b = x.shape[0]
        p = x.view(b, self.n_bits, 14, 1)

        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)

        z = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z)))))
        return self.group_sum(z)


@register("lgn-dense-2-y0-100-deepconv2-fuse3_tau10_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Y0BranchHuge200DeepConv2Fuse3FB(torch.nn.Module):
    n_bits = 100
    k_main = 1024
    k_y0 = 512

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}

        self.y0_fc1 = LogicDense(in_dim=self.n_bits, out_dim=3 * self.k_y0, device=device, parametrization_kwargs=pk)
        self.y0_fc2 = LogicDense(in_dim=3 * self.k_y0, out_dim=3 * self.k_y0, device=device, parametrization_kwargs=pk)
        self.y0_fc3 = LogicDense(in_dim=3 * self.k_y0, out_dim=3 * self.k_y0, device=device, parametrization_kwargs=pk)

        self.p_conv1 = LogicConv2d(
            in_dim=(13, 1),
            channels=self.n_bits,
            num_kernels=256,
            receptive_field_size=(3, 1),
            tree_depth=2,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        self.p_conv2 = LogicConv2d(
            in_dim=(11, 1),
            channels=256,
            num_kernels=512,
            receptive_field_size=(3, 1),
            tree_depth=2,
            stride=1,
            padding=0,
            device=device,
            parametrization_kwargs=pk,
        )

        p_conv_out = 512 * 9 * 1  

        self.p_fc1 = LogicDense(in_dim=p_conv_out, out_dim=3 * self.k_main, device=device, parametrization_kwargs=pk)
        self.p_fc2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, device=device, parametrization_kwargs=pk)
        self.p_fc3 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, device=device, parametrization_kwargs=pk)

    
        self.fuse1 = LogicDense(
            in_dim=3 * (self.k_main + self.k_y0),
            out_dim=3 * self.k_main,
            device=device,
            parametrization_kwargs=pk,
        )
        self.fuse2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, device=device, parametrization_kwargs=pk)
        self.fuse3 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, device=device, parametrization_kwargs=pk)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)

        y0 = x[:, :self.n_bits]                 
        p = x[:, self.n_bits:]                 

        y = self.y0_fc3(self.y0_fc2(self.y0_fc1(y0)))   
        b = p.shape[0]
        p2 = p.view(b, self.n_bits, 13, 1)              
        p2 = self.p_conv2(self.p_conv1(p2))             
        p2 = p2.view(p2.size(0), -1)                    

        p2 = self.p_fc3(self.p_fc2(self.p_fc1(p2)))     
        # fuse
        z = torch.cat([y, p2], dim=1)                   
        z = self.fuse3(self.fuse2(self.fuse1(z)))       

        return self.group_sum(z)
















### Previously Tested Models (subsample) ###



### Width Test
@register("lgn-dense-2-100", input_type="y-profile", task_type="classification")
class DenseLGN2_100(torch.nn.Sequential):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super(DenseLGN2_100, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2_100")
        param_kwargs = {"weight_init": "residual"}
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
        ]
        super(DenseLGN2_100, self).__init__(*layers, GroupSum(3, tau=10.0))


@register("lgn-dense-2-small-100", input_type="y-profile", task_type="classification")
class DenseLGN2Small100(DenseLGN2_100):
    n_bits = 100
    k = 512


@register("lgn-dense-2-medium-100", input_type="y-profile", task_type="classification")
class DenseLGN2Medium100(DenseLGN2_100):
    n_bits = 100
    k = 1024


@register("lgn-dense-2-large-100", input_type="y-profile", task_type="classification")
class DenseLGN2Large100(DenseLGN2_100):
    n_bits = 100
    k = 2048


@register("lgn-dense-2-huge-100", input_type="y-profile", task_type="classification")
class DenseLGN2Huge100(DenseLGN2_100):
    n_bits = 100
    k = 4096

### Depth Test
class _DenseLGN2DeepBase(torch.nn.Sequential):
    n_bits = 100
    k = 1024
    depth = 4  

    def __init__(self, thresholds):
        super(_DenseLGN2DeepBase, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in {self.__class__.__name__} (depth={self.depth})")
        param_kwargs = {"weight_init": "residual"}

        layers = [FixedBinarization(thresholds=thresholds)]

        layers.append(LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs))

        for _ in range(self.depth - 1):
            layers.append(LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs))

        super(_DenseLGN2DeepBase, self).__init__(*layers, GroupSum(3, tau=10.0))


@register("lgn-dense-2-deep4-100", input_type="y-profile", task_type="classification")
class DenseLGN2Deep4_100(_DenseLGN2DeepBase):
    n_bits = 100
    k = 1024
    depth = 4


@register("lgn-dense-2-deep5-100", input_type="y-profile", task_type="classification")
class DenseLGN2Deep5_100(_DenseLGN2DeepBase):
    n_bits = 100
    k = 1024
    depth = 5


@register("lgn-dense-2-deep6-100", input_type="y-profile", task_type="classification")
class DenseLGN2Deep6_100(_DenseLGN2DeepBase):
    n_bits = 100
    k = 1024
    depth = 6


@register("lgn-dense-2-deep7-100", input_type="y-profile", task_type="classification")
class DenseLGN2Deep7_100(_DenseLGN2DeepBase):
    n_bits = 100
    k = 1024
    depth = 7


@register("lgn-dense-2-deep8-100", input_type="y-profile", task_type="classification")
class DenseLGN2Deep8_100(_DenseLGN2DeepBase):
    n_bits = 100
    k = 1024
    depth = 8













def _conv3_fc_in(n_bits: int, c2: int, h: int, w: int) -> int:
    return c2 * h * w + n_bits


@register("lgn-conv-3-lb-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3LB100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


@register("lgn-conv-3-lb-200", input_type="y-profile-timing", task_type="classification")
class ConvLGN3LB200(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 200
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


# 2) Aspect change: widen conv channels (more kernels), keep bits=100 and FC same width
@register("lgn-conv-3-wideconv-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3WideConv100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        # only change: conv kernels 64->96, 128->192
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=96, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=96, num_kernels=192, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=192, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


# 3) Aspect change: deepen FC (add one extra FC layer), keep bits=100 and conv same
@register("lgn-conv-3-deepfc-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3DeepFC100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc4 = LogicDense(in_dim=3000, out_dim=3000, device=device)  # only change

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)  # only change
        return self.group_sum(x)


# 4) Aspect change: widen FC (increase k), keep bits=100 and conv same
@register("lgn-conv-3-widefc-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3WideFC100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        # only change: 3000 -> 6000 (still divisible by 3)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=6000, device=device)
        self.fc2 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.fc3 = LogicDense(in_dim=6000, out_dim=6000, device=device)

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


# 5) Aspect change: adjust GroupSum temperature (tau), keep everything else as base
@register("lgn-conv-3-tau5-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3Tau5_100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)

        # only change: tau 10 -> 5 (often sharper / less smoothing)
        self.group_sum = GroupSum(3, tau=5.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)





# =================================================
# 1 EXPERIMENTAL "accuracy amp" model:
# - deeper conv stack (3 convs) + wider FC
# - keeps raw input contract and valid shapes
# =================================================
@register("lgn-conv-3-amp-acc-100", input_type="y-profile-timing", task_type="classification")
class ConvLGN3AmpAcc100(torch.nn.Module):
    n_bits = 100

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 100
        self.binarization = FixedBinarization(thresholds=thresholds)

        # conv geometry:
        # (13,8) -> (11,6) -> (9,4) -> (7,2) with rf=3 each
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=96, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=96, num_kernels=192, receptive_field_size=3, tree_depth=2, device=device)
        self.conv3 = LogicConv2d(in_dim=(9, 4), channels=192, num_kernels=256, receptive_field_size=3, tree_depth=2, device=device)

        fc_in = _conv3_fc_in(self.n_bits, c2=256, h=7, w=2)
        # wider FC (still divisible by 3)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=9000, device=device)
        self.fc2 = LogicDense(in_dim=9000, out_dim=9000, device=device)
        self.fc3 = LogicDense(in_dim=9000, out_dim=9000, device=device)
        self.fc4 = LogicDense(in_dim=9000, out_dim=9000, device=device)  # extra depth like your best Dense-2 deep models

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return self.group_sum(x)


# =========================
# NEW MODELS (20 TOTAL)
# - Same conventions as your existing registry/classes
# - All use LearnableBinarization
# - Final projection always has out_dim divisible by 3 (we use 3*k)
# - Input dims match DenseLGN2: in_dim = n_bits * 14
# =========================

# -------------------------------------------------
# (1) 5 models with 100 bits (same structure as DenseLGN2, but learnable binarization)
# -------------------------------------------------



@register("lgn-dense-2-200", input_type="y-profile", task_type="classification")
class DenseLGN2_200(torch.nn.Sequential):
    n_bits = 200
    k = 2048

    def __init__(self, thresholds):
        super(DenseLGN2_200, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2_200")
        param_kwargs = {"weight_init": "residual"}
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
        ]
        super(DenseLGN2_200, self).__init__(*layers, GroupSum(3, tau=10.0))


# -------------------------------------------------
# (2) 5 models with 200 bits (same structure as DenseLGN2, but learnable binarization)
# -------------------------------------------------
# @register("lgn-dense-2-200", input_type="y-profile", task_type="classification")
# class DenseLGN2_200(torch.nn.Sequential):
#     n_bits = 200
#     k = 1024

#     def __init__(self, thresholds):
#         super(DenseLGN2_200, self).__init__()
#         print(f"Using {self.n_bits} bits for binarization in DenseLGN2_200")
#         param_kwargs = {"weight_init": "residual"}
#         layers = [
#             FixedBinarization(thresholds=thresholds),
#             LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
#             LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
#             LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
#         ]
#         super(DenseLGN2_200, self).__init__(*layers, GroupSum(3, tau=10.0))


@register("lgn-dense-2-small-200", input_type="y-profile", task_type="classification")
class DenseLGN2Small200(DenseLGN2_200):
    n_bits = 200
    k = 512


@register("lgn-dense-2-medium-200", input_type="y-profile", task_type="classification")
class DenseLGN2Medium200(DenseLGN2_200):
    n_bits = 200
    k = 1024


@register("lgn-dense-2-large-200", input_type="y-profile", task_type="classification")
class DenseLGN2Large200(DenseLGN2_200):
    n_bits = 200
    k = 4096


@register("lgn-dense-2-huge-200", input_type="y-profile", task_type="classification")
class DenseLGN2Huge200(DenseLGN2_200):
    n_bits = 200
    k = 8192




# -------------------------------------------------
# (4) 5 experimental 100-bit models (different structures/layer layouts)
#     Still: LearnableBinarization, correct input dim, final out_dim divisible by 3.
# -------------------------------------------------
@register("lgn-dense-2-bottleneck-100", input_type="y-profile", task_type="classification")
class DenseLGN2Bottleneck100(torch.nn.Module):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2Bottleneck100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)
        self.fc1 = LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc2 = LogicDense(in_dim=3 * self.k, out_dim=self.k, parametrization_kwargs=param_kwargs)      # bottleneck
        self.fc3 = LogicDense(in_dim=self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.group_sum(x)


@register("lgn-dense-2-hourglass-100", input_type="y-profile", task_type="classification")
class DenseLGN2Hourglass100(torch.nn.Module):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2Hourglass100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        self.fc1 = LogicDense(in_dim=self.n_bits * 14, out_dim=6 * self.k, parametrization_kwargs=param_kwargs)  # expand
        self.fc2 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)        # compress
        self.fc3 = LogicDense(in_dim=3 * self.k, out_dim=6 * self.k, parametrization_kwargs=param_kwargs)        # expand
        self.fc4 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)        # final to 3k

        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.group_sum(x)


@register("lgn-dense-2-splitconcat-100", input_type="y-profile", task_type="classification")
class DenseLGN2SplitConcat100(torch.nn.Module):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2SplitConcat100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        # Split features into two halves: (n_bits*7) and (n_bits*7)
        self.branch1 = LogicDense(in_dim=self.n_bits * 7, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.branch2 = LogicDense(in_dim=self.n_bits * 7, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        # After concat => 6k
        self.fuse1 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fuse2 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x1, x2 = torch.split(x, [self.n_bits * 7, self.n_bits * 7], dim=1)
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        x = torch.cat([b1, b2], dim=1)
        x = self.fuse1(x)
        x = self.fuse2(x)
        return self.group_sum(x)


@register("lgn-dense-2-widefirst-100", input_type="y-profile", task_type="classification")
class DenseLGN2WideFirst100(torch.nn.Module):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2WideFirst100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        self.fc1 = LogicDense(in_dim=self.n_bits * 14, out_dim=9 * self.k, parametrization_kwargs=param_kwargs)  # very wide
        self.fc2 = LogicDense(in_dim=9 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc3 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


@register("lgn-dense-2-residual-100", input_type="y-profile", task_type="classification")
class DenseLGN2Residual100(torch.nn.Module):
    n_bits = 100
    k = 1024

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN2Residual100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        self.fc_in = LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_a = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_b = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_out = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc_in(x)
        r = x
        x = self.fc_a(x)
        x = self.fc_b(x)
        x = x + r  # residual add (same shape: 3k)
        x = self.fc_out(x)
        return self.group_sum(x)

# =========================
# NEW LGN-DENSE-1 MODELS (20 TOTAL) — y-size
# - Same conventions as your existing registry/classes
# - All use LearnableBinarization (same as your DenseLGN1 baseline)
# - Input dim matches DenseLGN1: in_dim = n_bits * 2
# - Final projection always has out_dim divisible by 3 (we use 3*k)
# =========================

# -------------------------------------------------
# (1) 5 models with 100 bits (same DenseLGN1-style)
# -------------------------------------------------
@register("lgn-dense-1-100", input_type="y-size", task_type="classification")
class DenseLGN1_100(torch.nn.Sequential):
    n_bits = 100
    k = 256  # baseline scale (same idea as DenseLGN1)

    def __init__(self, thresholds):
        super(DenseLGN1_100, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1_100")
        param_kwargs = {"weight_init": "residual"}
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits * 2, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
        ]
        super(DenseLGN1_100, self).__init__(*layers, GroupSum(3, tau=1.0))


@register("lgn-dense-1-small-100", input_type="y-size", task_type="classification")
class DenseLGN1Small100(DenseLGN1_100):
    n_bits = 100
    k = 128


@register("lgn-dense-1-medium-100", input_type="y-size", task_type="classification")
class DenseLGN1Medium100(DenseLGN1_100):
    n_bits = 100
    k = 256


@register("lgn-dense-1-large-100", input_type="y-size", task_type="classification")
class DenseLGN1Large100(DenseLGN1_100):
    n_bits = 100
    k = 512


@register("lgn-dense-1-huge-100", input_type="y-size", task_type="classification")
class DenseLGN1Huge100(DenseLGN1_100):
    n_bits = 100
    k = 1024


# -------------------------------------------------
# (2) 5 models with 200 bits (same DenseLGN1-style)
# -------------------------------------------------
@register("lgn-dense-1-200", input_type="y-size", task_type="classification")
class DenseLGN1_200(torch.nn.Sequential):
    n_bits = 200
    k = 256

    def __init__(self, thresholds):
        super(DenseLGN1_200, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1_200")
        param_kwargs = {"weight_init": "residual"}
        layers = [
            FixedBinarization(thresholds=thresholds),
            LogicDense(in_dim=self.n_bits * 2, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
            LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs),
        ]
        super(DenseLGN1_200, self).__init__(*layers, GroupSum(3, tau=1.0))


@register("lgn-dense-1-small-200", input_type="y-size", task_type="classification")
class DenseLGN1Small200(DenseLGN1_200):
    n_bits = 200
    k = 128


@register("lgn-dense-1-medium-200", input_type="y-size", task_type="classification")
class DenseLGN1Medium200(DenseLGN1_200):
    n_bits = 200
    k = 256


@register("lgn-dense-1-large-200", input_type="y-size", task_type="classification")
class DenseLGN1Large200(DenseLGN1_200):
    n_bits = 200
    k = 512


@register("lgn-dense-1-huge-200", input_type="y-size", task_type="classification")
class DenseLGN1Huge200(DenseLGN1_200):
    n_bits = 200
    k = 1024


# -------------------------------------------------
# (3) 5 models with 100 bits, progressively deeper (each has +1 LogicDense vs previous)
#     Depth = number of LogicDense blocks after binarization.
#     Depths: 4, 5, 6, 7, 8
# -------------------------------------------------
class _DenseLGN1DeepBase(torch.nn.Sequential):
    n_bits = 100
    k = 256
    depth = 4  # override

    def __init__(self, thresholds):
        super(_DenseLGN1DeepBase, self).__init__()
        print(f"Using {self.n_bits} bits for binarization in {self.__class__.__name__} (depth={self.depth})")
        param_kwargs = {"weight_init": "residual"}

        layers = [FixedBinarization(thresholds=thresholds)]

        # first LogicDense consumes n_bits*2 -> 3k
        layers.append(LogicDense(in_dim=self.n_bits * 2, out_dim=3 * self.k, parametrization_kwargs=param_kwargs))

        # remaining (depth-1) blocks are 3k -> 3k
        for _ in range(self.depth - 1):
            layers.append(LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs))

        super(_DenseLGN1DeepBase, self).__init__(*layers, GroupSum(3, tau=1.0))


@register("lgn-dense-1-deep4-100", input_type="y-size", task_type="classification")
class DenseLGN1Deep4_100(_DenseLGN1DeepBase):
    n_bits = 100
    k = 256
    depth = 4


@register("lgn-dense-1-deep5-100", input_type="y-size", task_type="classification")
class DenseLGN1Deep5_100(_DenseLGN1DeepBase):
    n_bits = 100
    k = 256
    depth = 5


@register("lgn-dense-1-deep6-100", input_type="y-size", task_type="classification")
class DenseLGN1Deep6_100(_DenseLGN1DeepBase):
    n_bits = 100
    k = 256
    depth = 6


@register("lgn-dense-1-deep7-100", input_type="y-size", task_type="classification")
class DenseLGN1Deep7_100(_DenseLGN1DeepBase):
    n_bits = 100
    k = 256
    depth = 7


@register("lgn-dense-1-deep8-100", input_type="y-size", task_type="classification")
class DenseLGN1Deep8_100(_DenseLGN1DeepBase):
    n_bits = 100
    k = 256
    depth = 8


# -------------------------------------------------
# (4) 5 experimental 100-bit models (different layouts)
#     Still: LearnableBinarization, correct input dim (n_bits*2), final out_dim divisible by 3.
# -------------------------------------------------
@register("lgn-dense-1-bottleneck-100", input_type="y-size", task_type="classification")
class DenseLGN1Bottleneck100(torch.nn.Module):
    n_bits = 100
    k = 256

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1Bottleneck100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        self.fc1 = LogicDense(in_dim=self.n_bits * 2, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc2 = LogicDense(in_dim=3 * self.k, out_dim=self.k, parametrization_kwargs=param_kwargs)  # bottleneck
        self.fc3 = LogicDense(in_dim=self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc4 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=1.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.group_sum(x)


@register("lgn-dense-1-hourglass-100", input_type="y-size", task_type="classification")
class DenseLGN1Hourglass100(torch.nn.Module):
    n_bits = 100
    k = 256

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1Hourglass100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)
        self.fc1 = LogicDense(in_dim=self.n_bits * 2, out_dim=6 * self.k, parametrization_kwargs=param_kwargs)  # expand
        self.fc2 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)      # compress
        self.fc3 = LogicDense(in_dim=3 * self.k, out_dim=6 * self.k, parametrization_kwargs=param_kwargs)      # expand
        self.fc4 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)      # final to 3k

        self.group_sum = GroupSum(3, tau=1.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.group_sum(x)


@register("lgn-dense-1-splitconcat-100", input_type="y-size", task_type="classification")
class DenseLGN1SplitConcat100(torch.nn.Module):
    n_bits = 100
    k = 256

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1SplitConcat100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)

        # Split x into two halves: (n_bits) and (n_bits)
        self.branch1 = LogicDense(in_dim=self.n_bits, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.branch2 = LogicDense(in_dim=self.n_bits, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        # After concat => 6k
        self.fuse1 = LogicDense(in_dim=6 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fuse2 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=1.0)

    def forward(self, x):
        x = self.bin(x)
        x1, x2 = torch.split(x, [self.n_bits, self.n_bits], dim=1)
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        x = torch.cat([b1, b2], dim=1)
        x = self.fuse1(x)
        x = self.fuse2(x)
        return self.group_sum(x)


@register("lgn-dense-1-widefirst-100", input_type="y-size", task_type="classification")
class DenseLGN1WideFirst100(torch.nn.Module):
    n_bits = 100
    k = 256

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1WideFirst100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)
        self.fc1 = LogicDense(in_dim=self.n_bits * 2, out_dim=9 * self.k, parametrization_kwargs=param_kwargs)  # very wide
        self.fc2 = LogicDense(in_dim=9 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc3 = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=1.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)


@register("lgn-dense-1-residual-100", input_type="y-size", task_type="classification")
class DenseLGN1Residual100(torch.nn.Module):
    n_bits = 100
    k = 256

    def __init__(self, thresholds):
        super().__init__()
        print(f"Using {self.n_bits} bits for binarization in DenseLGN1Residual100")
        param_kwargs = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)
        self.fc_in = LogicDense(in_dim=self.n_bits * 2, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_a = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_b = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)
        self.fc_out = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=param_kwargs)

        self.group_sum = GroupSum(3, tau=1.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc_in(x)
        r = x
        x = self.fc_a(x)
        x = self.fc_b(x)
        x = x + r  # residual add (same shape: 3k)
        x = self.fc_out(x)
        return self.group_sum(x)


def _dense2_stack(thresholds, n_bits, k, depth, tau, param_kwargs=None, parametrization="raw"):
    layers = [FixedBinarization(thresholds=thresholds)]
    if param_kwargs is None:
        param_kwargs = {"weight_init": "residual"}
    layers.append(LogicDense(in_dim=n_bits * 14, out_dim=3 * k, parametrization=parametrization, parametrization_kwargs=param_kwargs))
    for _ in range(depth - 1):
        layers.append(LogicDense(in_dim=3 * k, out_dim=3 * k, parametrization=parametrization, parametrization_kwargs=param_kwargs))
    return torch.nn.Sequential(*layers, GroupSum(3, tau=tau))


@register("lgn-dense-2-huge-100-tau5_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge100Tau5FB(torch.nn.Module):
    n_bits = 100
    k = 4096

    def __init__(self, thresholds):
        super().__init__()
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=3, tau=5.0)

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-huge-200-deep5_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge100Deep5FB(torch.nn.Module):
    n_bits = 200
    k = 4096

    def __init__(self, thresholds):
        super().__init__()
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=5, tau=10.0)

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-huge-200-tau5_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge200Tau5FB(torch.nn.Module):
    n_bits = 200
    k = 8192

    def __init__(self, thresholds):
        super().__init__()
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=3, tau=5.0)

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-huge-200-deep4_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge200Deep4FB(torch.nn.Module):
    n_bits = 200
    k = 8192

    def __init__(self, thresholds):
        super().__init__()
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=4, tau=10.0)

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-huge-100-walsh-catalog_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge100WalshCatalogFB(torch.nn.Module):
    n_bits = 100
    k = 4096

    def __init__(self, thresholds):
        super().__init__()
        pk = {
            "weight_init": "residual-catalog",
            "residual_probability": 0.95,
            "forward_sampling": "soft",
            "temperature": 1.0,
        }
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=3, tau=10.0, param_kwargs=pk, parametrization="walsh")

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-huge-200-walsh-catalog_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Huge200WalshCatalogFB(torch.nn.Module):
    n_bits = 200
    k = 8192

    def __init__(self, thresholds):
        super().__init__()
        pk = {
            "weight_init": "residual-catalog",
            "residual_probability": 0.95,
            "forward_sampling": "soft",
            "temperature": 1.0,
        }
        self.net = _dense2_stack(thresholds, self.n_bits, self.k, depth=3, tau=10.0, param_kwargs=pk, parametrization="walsh")

    def forward(self, x):
        return self.net(x)


@register("lgn-dense-2-y0-branch-huge-100_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Y0BranchHuge100FB(torch.nn.Module):
    n_bits = 100
    k_main = 4096
    k_y0 = 1024

    def __init__(self, thresholds):
        super().__init__()
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        self.y0_fc1 = LogicDense(in_dim=self.n_bits, out_dim=3 * self.k_y0, parametrization_kwargs=pk)
        self.y0_fc2 = LogicDense(in_dim=3 * self.k_y0, out_dim=3 * self.k_y0, parametrization_kwargs=pk)
        self.p_fc1 = LogicDense(in_dim=self.n_bits * 13, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.p_fc2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.fuse1 = LogicDense(in_dim=3 * (self.k_main + self.k_y0), out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.fuse2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, :self.n_bits]
        p = x[:, self.n_bits:]
        y = self.y0_fc2(self.y0_fc1(y0))
        p = self.p_fc2(self.p_fc1(p))
        z = torch.cat([y, p], dim=1)
        z = self.fuse2(self.fuse1(z))
        return self.group_sum(z)



@register("lgn-dense-2-y0-branch-huge-200_fb", input_type="y-profile", task_type="classification")
class DenseLGN2Y0BranchHuge200FB(torch.nn.Module):
    n_bits = 100
    k_main = 4096
    k_y0 = 1024

    def __init__(self, thresholds):
        super().__init__()
        self.bin = FixedBinarization(thresholds=thresholds)
        pk = {"weight_init": "residual"}
        self.y0_fc1 = LogicDense(in_dim=self.n_bits, out_dim=3 * self.k_y0, parametrization_kwargs=pk)
        self.y0_fc2 = LogicDense(in_dim=3 * self.k_y0, out_dim=3 * self.k_y0, parametrization_kwargs=pk)
        self.p_fc1 = LogicDense(in_dim=self.n_bits * 13, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.p_fc2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.fuse1 = LogicDense(in_dim=3 * (self.k_main + self.k_y0), out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.fuse2 = LogicDense(in_dim=3 * self.k_main, out_dim=3 * self.k_main, parametrization_kwargs=pk)
        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, :self.n_bits]
        p = x[:, self.n_bits:]
        y = self.y0_fc2(self.y0_fc1(y0))
        p = self.p_fc2(self.p_fc1(p))
        z = torch.cat([y, p], dim=1)
        z = self.fuse2(self.fuse1(z))
        return self.group_sum(z)











@register("lgn-dense-2-residual-block-huge-100_fb", input_type="y-profile", task_type="classification")
class DenseLGN2ResidualBlockHuge100FB(torch.nn.Module):
    n_bits = 100
    k = 4096

    def __init__(self, thresholds):
        super().__init__()
        pk = {"weight_init": "residual"}
        self.bin = FixedBinarization(thresholds=thresholds)
        self.fc_in = LogicDense(in_dim=self.n_bits * 14, out_dim=3 * self.k, parametrization_kwargs=pk)
        self.fc_a = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=pk)
        self.fc_b = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=pk)
        self.fc_out = LogicDense(in_dim=3 * self.k, out_dim=3 * self.k, parametrization_kwargs=pk)
        self.group_sum = GroupSum(3, tau=10.0)

    def forward(self, x):
        x = self.bin(x)
        x = self.fc_in(x)
        r = x
        x = self.fc_b(self.fc_a(x))
        x = x + r
        x = self.fc_out(x)
        return self.group_sum(x)


def _conv3_fc_in(n_bits: int, c2: int, h: int, w: int) -> int:
    return c2 * h * w + n_bits






@register("lgn-conv-3-wideconv-200_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3WideConv200FB(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=96, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=96, num_kernels=192, receptive_field_size=3, tree_depth=2, device=device)
        fc_in = _conv3_fc_in(self.n_bits, 192, 9, 4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc3(self.fc2(self.fc1(z)))
        return self.group_sum(z)


@register("lgn-conv-3-tree3-200_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3Tree3_200FB(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=3, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=3, device=device)
        fc_in = _conv3_fc_in(self.n_bits, 128, 9, 4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device)
        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc3(self.fc2(self.fc1(z)))
        return self.group_sum(z)


@register("lgn-conv-3-multiscale-200_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3MultiScale200FB(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        self.a1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.a2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)
        self.b1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=128, receptive_field_size=5, tree_depth=2, device=device)
        fc_in = (128 * 9 * 4 + 128 * 9 * 4) + self.n_bits
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=6000, device=device)
        self.fc2 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.fc3 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        za = self.a2(self.a1(p))
        zb = self.b1(p)
        za = za.view(za.size(0), -1)
        zb = zb.view(zb.size(0), -1)
        z = torch.cat((y0, za, zb), dim=1)
        z = self.fc3(self.fc2(self.fc1(z)))
        return self.group_sum(z)


@register("lgn-conv-3-residualfc-200_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3ResidualFC200FB(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)
        fc_in = _conv3_fc_in(self.n_bits, 128, 9, 4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=6000, device=device)
        self.fc2 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.fc3 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.fc4 = LogicDense(in_dim=6000, out_dim=6000, device=device)
        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc1(z)
        r = z
        z = self.fc3(self.fc2(z))
        z = z + r
        z = self.fc4(z)
        return self.group_sum(z)

@register("lgn-conv-3-residual-200-tau5_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3LB200Tau5FBr(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64,
                                 receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128,
                                 receptive_field_size=3, tree_depth=2, device=device)

        pk = {"weight_init": "residual"}  # <-- this changed for random vs residual

        fc_in = _conv3_fc_in(self.n_bits, 128, 9, 4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)  # <-- change
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)   # <-- change
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)   # <-- change

        self.group_sum = GroupSum(3, tau=5.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc3(self.fc2(self.fc1(z)))
        return self.group_sum(z)


@register("lgn-conv-3-residual-200_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3LB200r(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        device = "cuda"
        super().__init__()
        self.n_bits = 200
        self.binarization = FixedBinarization(thresholds=thresholds)

        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64,
                                 receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128,
                                 receptive_field_size=3, tree_depth=2, device=device)

        pk = {"weight_init": "residual"}  
        fc_in = _conv3_fc_in(self.n_bits, c2=128, h=9, w=4)
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)  
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)   
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)   

        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"
        x = self.binarization(x)

        y0 = x[:, 0:self.n_bits]
        profile = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)

        x = self.conv1(profile)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = torch.cat((y0, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.group_sum(x)
@register("lgn-conv-3-deepfc-200-random-_fb", input_type="y-profile-timing", task_type="classification")
class ConvLGN3DeepFC200FBr(torch.nn.Module):
    n_bits = 200

    def __init__(self, thresholds):
        super().__init__()
        device = "cuda"
        self.n_bits = 200
        self.bin = FixedBinarization(thresholds=thresholds)
        self.conv1 = LogicConv2d(in_dim=(13, 8), channels=self.n_bits, num_kernels=64, receptive_field_size=3, tree_depth=2, device=device)
        self.conv2 = LogicConv2d(in_dim=(11, 6), channels=64, num_kernels=128, receptive_field_size=3, tree_depth=2, device=device)
        fc_in = _conv3_fc_in(self.n_bits, 128, 9, 4)
        pk = {"weight_init": "random"}
        self.fc1 = LogicDense(in_dim=fc_in, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc2 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc3 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.fc4 = LogicDense(in_dim=3000, out_dim=3000, device=device, parametrization_kwargs=pk)
        self.group_sum = GroupSum(3, tau=10.0, device=device)

    def forward(self, x):
        x = self.bin(x)
        y0 = x[:, 0:self.n_bits]
        p = x[:, self.n_bits:].view(-1, self.n_bits, 13, 8)
        z = self.conv2(self.conv1(p))
        z = z.view(z.size(0), -1)
        z = torch.cat((y0, z), dim=1)
        z = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return self.group_sum(z)




































# @register("towards-model-4", input_type="y-profile-timing", task_type="classification")    

""" from the original papers
(i) Model 1: cluster y-size. This model used the cluster position on a flat module (y0), and cluster y-size,
which is the number of pixel rows with nonzero charge deposited after 4 nanoseconds. This model has
two input features: y0 position (1 feature) and cluster y-size (1 feature). This model consists of one
dense layer with 128 neurons and 384 parameters, followed by one dense layer with 3 neurons and 387
parameters. The model provides a test of performance with minimal information provided to the
neural network.
(ii) Model 2: cluster y-profile. This model has fourteen input features: y0 position (1 feature) and cluster
y-profile (13 features corresponding to 13 pixel rows). Cluster y-profile represents the amount of
charge collected in each row of pixels after 4 nanoseconds. The model consists of one dense layer with
128 neurons and 1920 parameters, followed by one dense layer with 3 neurons and 387 parameters.
(iii) Model 3: cluster y-profile with timing information. The third and most complex model takes as input
the y0 position (1 feature) and the cluster y-profile distribution at eight time slices (13 × 8 features),
which represents the amount of charge collected in each row of pixels evaluated at eight intervals of
200 picoseconds. The earliest time slices contain the most useful information, as most charge
deposition occurs at the beginning of the cluster time evolution. This model uses a convolutional
neural network to pass a time-lapse picture of the cluster charge to the network. Cluster y-profile inputs
were passed through two two-dimensional convolutional layers (Conv2D), with 16 and 64 filters,
respectively, using ReLU activations to introduce non-linearity [23]. The shape of the kernels was 3 × 3,
and strides was 1 × 1. The output of the Conv2D layers was flattened and concatenated with the y0
input. This was then passed through a dense layer with 32 neurons, using dropout of 0.1. The final
model contains 83 331 parameters.

The cluster y-profile is calculated
for each event as the sensor output at 4000 ps summed over pixel rows (x) to project the integrated charge
of the cluster onto the y-axis. Additionally, y0, the azimuthal position of the particle’s incident position on
the sensor array in the global detector coordinates, is passed as input. The inter-dependence of the cluster y-
profile, y0, and pT was examined in [1]. This network, therefore taking in 14 input values (y0 and 13 values
from cluster y-profile), contains one dense hidden layer with 128 neurons and 1920 parameters, followed
by one dense output layer with 3 neurons and 387 parameters. A softmax activation is used to generate
classification probabilities between 0 and 1, and each event is assigned the classification label corresponding
to the highest probability.
"""
