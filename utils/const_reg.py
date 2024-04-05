# train에서 임포트해야하는데 미제공된 파일이라 내가 작성

import torch
import torch.nn.functional as F

def consistency_cost(model, teacher_model, imgs):

    model_preds=model(imgs)
    teacher_preds = teacher_model(imgs)

    mse_loss = F.mse_loss(model_preds, teacher_preds)

    return mse_loss
