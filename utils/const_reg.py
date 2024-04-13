# train에서 임포트해야하는데 미제공된 파일이라 내가 작성

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


def consistency_cost(model, teacher_model, imgs):
    # teacher_model = weak aug(random rotation, crop)
    # student_model = strong aug (brightness change)
    strong_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.5)  # 밝기 변경
        # 여기에 필요한 다른 morphological 변환을 추가할 수 있습니다.
    ])

    weak_aug = transforms.Compose([
        transforms.RandomRotation(),  # 무작위 회전
        transforms.RandomCrop()  # 무작위 크롭, 크기는 필요에 따라 설정
        # 필요한 경우 여기에 추가 증강을 넣을 수 있습니다.
    ])

    # 이미지에 증강 적용
    weak_aug_imgs = weak_aug(imgs)
    strong_aug_imgs = strong_aug(imgs)

    # 모델에 증강된 이미지 전달
    model_preds = model(strong_aug_imgs)
    teacher_preds = teacher_model(weak_aug_imgs)

    # MSE 손실 계산
    mse_loss = F.cross_entropy(model_preds, teacher_preds)

    return mse_loss
