"""ResNet18 迁移学习模型及图像预处理。"""

from torchvision import models, transforms
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """以 ImageNet ResNet18 初始化，并替换为本实验类别数的分类头。"""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if pretrained:
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
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
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
