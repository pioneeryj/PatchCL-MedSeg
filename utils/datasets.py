# train 에서 불러와야하는데 미제공 파일이라 내가 작성함
# Dataloader() 안에 들어올 dataset 
from torch.utils.data import Dataset 

class LabData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root=root_dir
        print(root_dir)
        self.image_dir=os.path.join(root_dir, 'img_slices')
        self.label_dir=os.path.join(root_dir, 'label_sices')
        self.image_namelist = self.loadName(root_dir)

    def loadName(self, root_dir):
        with open(os.path.join)
