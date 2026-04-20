import torch
import torch.nn as nn
import torchvision.models as tvm

from .config import NUM_CLASSES



# region DilatedSegCNN
class DilatedSegCNN(nn.Module):
    """
    Custom segmentation CNN with dilated encoder blocks and transposed-conv
    decoder with residual skip connections.

    Encoder dilations: 1 → 2 → 4 → 8  (captures multi-scale texture without
    pooling-induced spatial loss in the early layers).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = self._block(in_channels,  64,  dilation=1)
        self.enc2 = self._block(64,  128, dilation=2)
        self.enc3 = self._block(128, 256, dilation=4)
        self.enc4 = self._block(256, 512, dilation=8)
        self.dec3 = self._upblock(512, 256)
        self.dec2 = self._upblock(256, 128)
        self.dec1 = self._upblock(128,  64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _block(self, in_c: int, out_c: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def _upblock(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x  = self.dec3(x4) + x3   # residual skip
        x  = self.dec2(x)  + x2
        x  = self.dec1(x)  + x1
        return self.classifier(x)
# endregion

# region UNet++ with pretrained ResNet34 encoder
def build_unetpp(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build a UNet++ model with a ResNet34 encoder pre-trained on ImageNet.

    The encoder's first conv layer is adapted for 1-channel (grayscale) input
    by averaging the 3-channel weights along the channel dimension.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for UNet++. "
            "Install with: pip install segmentation-models-pytorch"
        ) from exc

    model = smp.UnetPlusPlus(
        encoder_name    = "resnet34",
        encoder_weights = None,       # weights loaded manually below
        in_channels     = 1,
        classes         = num_classes,
        activation      = None,
    )

    # Adapt pretrained ImageNet weights for 1-channel input
    enc_w = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1).state_dict()
    enc_w["conv1.weight"] = enc_w["conv1.weight"].mean(dim=1, keepdim=True)

    model_dict = model.encoder.state_dict()
    matched    = {k: v for k, v in enc_w.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.encoder.load_state_dict(model_dict, strict=False)
    print(f"  UNet++ encoder: {len(matched)}/{len(model_dict)} ImageNet weights loaded")
    return model
# endregion


# region Unified checkpoint loader
def load_checkpoint(
    model_cfg: dict,
    device:    torch.device | None = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Build the right model architecture and load weights from *model_cfg["path"]*.

    Parameters:
    model_cfg : entry from ``config.MODELS``
    device    : torch device; defaults to CUDA if available
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = model_cfg["type"]
    if model_type == "dilated":
        model = DilatedSegCNN(in_channels=1, num_classes=num_classes)
    elif model_type == "unetpp":
        model = build_unetpp(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Expected 'dilated' or 'unetpp'.")

    state = torch.load(model_cfg["path"], map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
# endregion
