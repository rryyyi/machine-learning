"""ResNet 迁移学习模型及图像预处理。"""

from torchvision import models, transforms
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(
    num_classes: int,
    pretrained: bool = True,
    fine_tune_all: bool = False,
    model_name: str = "resnet18",
) -> nn.Module:
    """创建 ImageNet 初始化的 ResNet18/50，并替换分类头。"""

    model_factories = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    }
    if model_name not in model_factories:
        raise ValueError(f"暂不支持模型 {model_name!r}，可选：{tuple(model_factories)}")

    model_factory, weights_type = model_factories[model_name]
    model = model_factory(weights=weights_type if pretrained else None)

    if pretrained and not fine_tune_all:
        # 小样本下先保留通用低层特征，只微调高层和新分类头，降低过拟合风险。
        for parameter in model.parameters():
            parameter.requires_grad = False
        for layer in (model.layer3, model.layer4):
            for parameter in layer.parameters():
                parameter.requires_grad = True

    feature_count = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(feature_count, num_classes),
    )
    return model


def make_train_transform(image_size: int = 224):
    """通过轻量增强提升小数据集的泛化能力。"""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_eval_transform(image_size: int = 224):
    """验证和测试使用确定性缩放/裁剪，保证推理一致。"""

    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
