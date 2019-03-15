import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class NIC_InverseL1Loss(torch.nn.Module):
    def forward(self, output, target):
        l1_voxel = torch.mean(torch.abs(output - target), dim=1)
        return 1.0 - (torch.mean(l1_voxel) / torch.max(l1_voxel))


class NIC_autodenoiser_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.ND_loss_func = torch.nn.MSELoss()
        self.ND_loss_func = torch.nn.L1Loss()
        self.DS_loss_func = NIC_binary_xent_gdl(type_weight='Simple')
        self.NI_loss_func = NIC_InverseL1Loss() # Zero noise error

    def forward(self, output_pack, target_pack):
        if isinstance(output_pack, list) and isinstance(target_pack, list):
            output_DS, output_ND, output_D = output_pack[0], output_pack[1], output_pack[2]
            target_DS, target_ND = target_pack[0], target_pack[1]

            DS_loss = 10.0 * self.DS_loss_func(output_DS, target_DS)
            ND_loss = self.ND_loss_func(output_ND, target_ND)

            return DS_loss + ND_loss
        else:
            output_DS = output_pack
            target_DS = target_pack
            DS_loss = self.DS_loss_func(output_DS, target_DS)
            return DS_loss


class NIC_MSE_reg_l1_fft(torch.nn.Module):
    def __init__(self, w_reg=1.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.w_reg = w_reg

    def forward(self, output, target):
        t_fft = torch.rfft(torch.round(output), len(output.shape) - 2, normalized=True)
        l1 = torch.sum(torch.pow(torch.sum(torch.pow(t_fft, 2.0), dim=-1) , 0.5)) / output.numel()
        return self.mse(output, target) + self.w_reg * l1

class NIC_crossentropyloss(torch.nn.Module):
    def __init__(self, weights=(0.5, 0.5), device=torch.device('cuda')):
        super().__init__()
        self.w = torch.tensor(weights).to(device)

    def forward(self, output, target):
        output = torch.log(torch.clamp(output, 1E-7, 1.0 - 1E-7))
        target = torch.squeeze(target, dim=1).long()
        return F.cross_entropy(output, target, weight=self.w)

def nic_binary_l1_er(y_pred, y_true):
    """
    Early stopping for ISLES 2018
    """
    return nic_binary_l1loss(y_pred, y_true) + nic_binary_error_rate(y_pred, y_true)

def nic_binary_crossentropy(y_pred, y_true):
    y_true_binary = torch.cat([torch.abs(y_true - 1), y_true], dim=1).float()
    y_pred = torch.clamp(y_pred, 1E-7, 1. - 1E-7)
    return torch.mean(-torch.sum(y_true_binary * torch.log(y_pred), dim=1))

def nic_binary_hinge(y_pred, y_true):
    y_true_binary = torch.cat([torch.abs(y_true - 1), y_true], dim=1).float()

    pos = torch.sum(y_true_binary * y_pred, dim=1)
    neg, _ = torch.max((1. - y_true_binary) * y_pred, dim=1)
    hinge = torch.max(torch.zeros_like(pos), neg - pos + 1.0)
    return torch.mean(hinge)

def nic_binary_jaccard(y_pred, y_true, smooth=100.0):
    y_true_binary = torch.cat([torch.abs(y_true - 1.0), y_true], dim=1).float()
    y_pred = torch.clamp(y_pred, 1E-7, 1. - 1E-7)

    intersection = torch.sum(torch.abs(y_true_binary * y_pred), dim=1)
    sum_ = torch.sum(torch.abs(y_true_binary) + torch.abs(y_pred), dim=1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return torch.mean(1.0 - jac) * smooth

def nic_binary_kl_divergence(y_pred, y_true):
    y_true_binary = torch.cat([torch.abs(y_true - 1.), y_true], dim=1).float()
    y_pred = torch.log(torch.clamp(y_pred, 1E-7, 1. - 1E-7))
    return F.kl_div(y_pred, y_true_binary)

def nic_binary_mseloss(y_pred, y_true):
    y_true_binary = torch.cat([torch.abs(y_true - 1.), y_true], dim=1).float()
    y_pred = torch.clamp(y_pred, 1E-7, 1. - 1E-7)
    return F.mse_loss(y_pred, y_true_binary)

def nic_binary_l1loss(y_pred, y_true):
    y_true_binary = torch.cat([torch.abs(y_true - 1.), y_true], dim=1).float()
    y_pred = torch.clamp(y_pred, 1E-7, 1. - 1E-7)
    return F.l1_loss(y_pred, y_true_binary)

def nic_binary_bceloss(y_pred, y_true):
    """
    Wrapper for torch BCELoss that adapts the shape and content
    """
    y_true_binary = torch.cat([torch.abs(y_true - 1.0), y_true], dim=1).float()
    y_pred = torch.clamp(y_pred, min=1E-7, max=1.0 - 1E-7)
    return F.binary_cross_entropy(y_pred, y_true_binary)

def nic_binary_accuracy(y_pred, y_true, class_dim=1):
    """
    from Keras: K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    """
    y_true = torch.squeeze(y_true.long(), dim=class_dim)
    y_pred_categorical = torch.argmax(y_pred, dim=class_dim)
    return torch.mean(torch.eq(y_true, y_pred_categorical).float())

def nic_binary_error_rate(y_pred, y_true, class_dim=1):
    return 1.0 - nic_binary_accuracy(y_pred, y_true, class_dim)

def nic_lesion_dice_loss(y_pred, y_true, smooth=100.0):
    y_pred = y_pred[:, 1, ...].unsqueeze(1) # only lesion probabilities
    dice_numerator = 2.0 * torch.sum(y_pred * y_true, dim=1)
    dice_denominator = torch.sum(y_pred, dim=1) + torch.sum(y_true, dim=1)
    dice_score = (dice_numerator + smooth) / (dice_denominator + smooth)
    return torch.mean(1.0 - dice_score) * smooth

def nic_binary_dice(output, target, smooth=100.0):
    """
    Computing mean-class Dice similarity.
    """
    target = torch.cat([1.0 - target, target], dim=1).float()
    dice_numerator = 2.0 * torch.sum(output * target, dim=1)
    dice_denominator = torch.sum(output, dim=1) + torch.sum(target, dim=1)
    dice_score = (dice_numerator + smooth) / (dice_denominator + smooth)
    return torch.mean(1.0 - dice_score) * smooth


class NIC_binary_xent_gdl(torch.nn.Module):
    def __init__(self, xent_weights=(0.5, 0.5), type_weight='Square', device=torch.device('cuda')):
        super().__init__()
        self.gdl = NIC_GDL(device=device, type_weight=type_weight)
        self.xent = NIC_crossentropyloss(device=device, weights=xent_weights)

    def forward(self, output, target):
        return self.gdl(output, target) + self.xent(output, target)

class NIC_binary_xent_dice(torch.nn.Module):
    def __init__(self, xent_weights=(0.5, 0.5), device=torch.device('cuda')):
        super().__init__()
        self.xent = NIC_crossentropyloss(device=device, weights=xent_weights)

    def forward(self, output, target):
        return nic_binary_dice(output, target) + self.xent(output, target)

class NIC_binary_exp_log_loss(torch.nn.Module):
    def __init__(self, w_dice=0.8, w_cross=0.2, gamma=0.3):
        super().__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
        self.gamma = gamma

    def forward(self, output_in, target_in):
        target = torch.cat([1 - target_in, target_in], dim=1).float()
        output = torch.clamp(output_in, min=1E-7, max=1.0 - 1E-7)

        # Dice term
        sum_dims = [0] + list(range(2, len(target.shape)))
        dice = (2.0 * torch.sum(target * output, dim=sum_dims) + 1.0) / (torch.sum(target + output, dim=sum_dims) + 1.0)
        Ldice = torch.mean(torch.pow(-torch.log(dice), self.gamma))

        # Weighted crossentropy term
        wl = torch.pow(torch.sum(target) / torch.sum(target, dim=[0] + list(range(2, len(target.shape)))), 0.5)
        cross_0 = wl[0] * torch.pow(-torch.log(output[:, 0, ...]), self.gamma)
        cross_1 = wl[1] * torch.pow(-torch.log(output[:, 1, ...]), self.gamma)
        Lcross = torch.mean(cross_0 + cross_1)

        return self.w_dice * Ldice + self.w_cross * Lcross



class NIC_GDL(torch.nn.Module):
    def __init__(self, device, type_weight='Square'):
        super(NIC_GDL, self).__init__()
        assert type_weight in {'Square', 'Simple', 'Uniform'}
        self.device = device
        self.wtype = type_weight

    def forward(self, pred, gt, type_weight='Square'):
        reduce_dims = (0, ) + tuple(range(2, len(gt.shape)))  # 0 and 1 are batch size
        gt_binary = torch.cat([1.0 - gt, gt], dim=1).float()

        # 1st compute weights
        if self.wtype == 'Square':
            weights = torch.reciprocal(torch.pow(torch.sum(gt_binary, dim=reduce_dims), 2.0))
        elif self.wtype == 'Simple':
            weights = torch.reciprocal(torch.sum(gt_binary, dim=reduce_dims))
        elif self.wtype == 'Uniform':
            weights = torch.ones_like(torch.sum(gt_binary, dim=reduce_dims))
        else:
            raise ValueError("The variable type_weight \"{}\" is not defined.".format(type_weight))
        new_weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * torch.max(new_weights), weights)

        # Compute score
        num = torch.sum(weights * torch.sum(torch.mul(pred, gt_binary), dim=reduce_dims))
        den = torch.sum(weights * torch.sum(torch.add(pred, gt_binary), dim=reduce_dims))
        gds = 2.0 * (num / den)
        if torch.isnan(gds):
            gds = torch.ones_like(gds)

        # Return loss
        return 1.0 - gds

class NIC_binary_asymsimilarity_loss(torch.nn.Module):
    def __init__(self, beta=1.5, device=torch.device('cuda')):
        super().__init__()
        self.b = torch.tensor(beta).to(device)

    def forward(self, output, target):
        target = torch.squeeze(target, dim=1)
        output = torch.clamp(output[:, 1, ...], min=1E-7, max=1.0)

        b2, b2_1 = torch.pow(self.b, 2.0), torch.pow(self.b, 2.0) + 1.0

        num = b2_1 * torch.sum(output * target) + 1.0
        den = b2_1 * torch.sum(output * target) + b2 * torch.sum((1.0 - output) * target) + \
              torch.sum(output * (1.0 - target)) + 1.0

        return torch.div(num, den)

class NIC_binary_focal_loss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Constructor for the binary focal loss
        :param gamma
        :param alpha: alpha is weight for class 0 (background)
        """
        super().__init__()
        self.g = gamma
        self.a = alpha

    def forward(self, y_pred, y_true):
        y_true_bin = torch.cat([1.0 - y_true, y_true], dim=1).float()
        y_pred = torch.clamp(y_pred, min=1E-7, max=1.0 - 1E-7)

        mask0, mask1  = y_true_bin[:, 0, ...], y_true_bin[:, 1, ...] # background mask (is 1 if bg, 0 if fg)
        s0, s1 = y_pred[:, 0, ...], y_pred[:, 1, ...] # s0 = 1 - s1

        fl_0 = torch.mul(mask0, self.a * torch.pow(1. - s0, self.g) * torch.log(s0))
        fl_1 = torch.mul(mask1, (1 - self.a) * torch.pow(1. - s1, self.g) * torch.log(s1))

        return torch.mean(-1.0 * (fl_0 + fl_1))