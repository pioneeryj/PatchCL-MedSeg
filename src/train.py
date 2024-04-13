import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import math
import time
import sys
import torch.nn.functional as F
sys.path.append('C:/Users/herbw/OneDrive/#IIPL/2024_IEIE/PatchCL-MedSeg')
#from utils.datasets import LabData,UnlabData
from torch.utils.data import DataLoader
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
from dataloaders.dataset import LabData,UnlabData
from utils.queues import Embedding_Queues
from utils.CELOSS import CE_loss
from utils.patch_utils import _get_patches
from utils.aug_utils import batch_augment
from utils.get_embds import get_embeddings
from utils.const_reg import consistency_cost
from utils.plg_loss import PCGJCL
from utils.torch_poly_lr_decay import PolynomialLRDecay

if __name__=="__main__":
	dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	stochastic_approx = StochasticApprox(4,0.5,0.8)

	model = Network()
	teacher_model = Network()

	#Turning off gradients for teacher model
	for param in teacher_model.parameters():
		param.requires_grad=False
		#Esuring mothe the models have same weight
	teacher_model.load_state_dict(model.state_dict())
	model.contrast=False
	teacher_model.contrast = False

	model = nn.DataParallel(model)
	model = model.to(dev)
	teacher_model = nn.DataParallel(teacher_model)
	teacher_model=teacher_model.to(dev)

	embd_queues = Embedding_Queues(4)

	cross_entropy_loss=CE_loss()
	metrics=[smp.utils.metrics.IoU(threshold=0.5)]

	optimizer_pretrain=torch.optim.Adam(model.parameters(),lr=0.001)
	optimizer_ssl=torch.optim.SGD(model.parameters(),lr=0.007)
	# 내가 추가
	optimizer_contrast=torch.optim.Adam(model.parameters(),lr=0.001)


	scheduler = PolynomialLRDecay(optimizer=optimizer_ssl, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)
	contrastive_batch_size = 128


	labeled_dataset = LabData()
	unlabeled_dataset = UnlabData()

	def my_collate_fn(samples):
		collate_images = []
		collate_masks = []

		target_height = 400
		target_width = 400

		for sample in samples:
			# 이미지를 PyTorch 텐서로 변환
			image_tensor = torch.tensor(sample['image'], dtype=torch.float)
			# 이미지 패딩 계산 및 추가
			height_pad = (target_height - image_tensor.shape[0]) // 2
			width_pad = (target_width - image_tensor.shape[1]) // 2
			# top, bottom, left, right 패딩
			pad_image = (width_pad, target_width - image_tensor.shape[1] - width_pad,
						height_pad, target_height - image_tensor.shape[0] - height_pad)
			padded_image = F.pad(image_tensor, pad_image, "constant", 0)
			collate_images.append(padded_image)

			# 마스크를 PyTorch 텐서로 변환
			mask_tensor = torch.tensor(sample['mask'], dtype=torch.float)
			# 마스크 패딩 계산 및 추가
			pad_mask = (width_pad, target_width - mask_tensor.shape[1] - width_pad,
						height_pad, target_height - mask_tensor.shape[0] - height_pad)
			padded_mask = F.pad(mask_tensor, pad_mask, "constant", 0)
			collate_masks.append(padded_mask)

		return {'image': torch.stack(collate_images), 'mask': torch.stack(collate_masks)}


	labelled_dataloader = DataLoader(labeled_dataset,batch_size=8, collate_fn=my_collate_fn, shuffle=True) #이것을 거치면 (img, mask 형태로 나오게 되는거)
	unlabeled_dataloader = DataLoader(unlabeled_dataset,batch_size=8,shuffle=True)

	#CONTRASTIVE PRETRAINING (warm up)
	#torch.autograd.set_detect_anomaly(True)
	for c_epochs in range(100): #100 epochs supervised pre training
		step=0
		min_loss = math.inf
		epoch_loss=0
		#print('Epoch ',c_epochs)

		for batch in labelled_dataloader:

			imgs = batch['image']
			masks = batch['mask']


			t1=time.time()
			with torch.no_grad():

				#Send psudo masks & imgs to cpu
				p_masks=masks
				imgs = imgs

				#get classwise patch list
				patch_list = _get_patches(imgs,p_masks)
				
				#stochastic approximation filtering and threshold update
				#qualified_patch_list = stochastic_approx.update(patch_list)
				qualified_patch_list = patch_list

				#make augmentations for teacher model
				augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)

				
				#convert to tensor
				aug_tensor_patch_list=[]
				qualified_tensor_patch_list=[]
				for i in range(len(augmented_patch_list)):
					if augmented_patch_list[i] is not None:
						aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
						qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
					else:
						aug_tensor_patch_list.append(None)
						qualified_tensor_patch_list.append(None)
			

			#get embeddings of qualified patches through student model
			model=model.train()
			model.module.contrast=True
			student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

			#get embeddings of augmented patches through teacher model
			teacher_model.train()
			teacher_model.module.contrast = True
			teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

			#enqueue these
			embd_queues.enqueue(teacher_embedding_list)

			#calculate PCGJCL loss
			PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)

			#calculate supervied loss
   
			#print(f'img shape: {imgs.shape} and mask shape: {masks.shape}') #torch.Size([8, 384, 276])
			imgs= imgs.unsqueeze(1)
			imgs = imgs.repeat(1,3,1,1)
			imgs, masks =imgs.to(dev), masks.to(dev)
			print(f'imgs shape {imgs.shape}')
			model.module.contrast=False
			out = model(imgs)
			print(f'out shape {out.shape}')
			print(f'mask shape {masks.shape}')
			supervised_loss = cross_entropy_loss(out,masks)

			#total loss
			loss = supervised_loss + 0.5*PCGJCL_loss

			epoch_loss+=loss
			
			#backpropagate
			loss.backward()
			optimizer_contrast.step()


			for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
				param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

			#Extras
			t2=time.time()
			print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
			step+=1
		if epoch_loss < min_loss:
			torch.save(model,'./best_contrast.pth')


	for c_epochs in range(200): #200 epochs supervised SSL
		step=0
		min_loss = math.inf
		epoch_loss=0
		#print('Epoch ',c_epochs)

		labeled_iterator = iter(labelled_dataloader)
		for imgs in unlabeled_dataloader:

			t1=time.time()
			with torch.no_grad():

				#send imgs to dev
				imgs = imgs.to(dev)
				
				#set model in Eval mode
				model = model.eval()

				#Get pseudo masks
				model.module.contrast=False
				p_masks = model(imgs)

				#Send psudo masks & imgs to cpu
				p_masks=masks
				p_masks = p_masks.to('cpu').detach()
				imgs = imgs.to('cpu').detach()

				#Since we use labeled data for PCGJCL as well
				imgs2, masks2 = labeled_iterator.next()

				#concatenating unlabeled and labeled sets
				p_masks = torch.cat([p_masks,masks2],dim=0)
				imgs = torch.cat([imgs,imgs2],dim=0)

				#get classwise patch list
				patch_list = _get_patches(imgs,p_masks)
				
				#stochastic approximation filtering and threshold update
				qualified_patch_list = stochastic_approx.update(patch_list)


				#make augmentations for teacher model
				augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)

				#convert to tensor
				aug_tensor_patch_list=[]
				qualified_tensor_patch_list=[]
				for i in range(len(augmented_patch_list)):
					if augmented_patch_list[i] is not None:
						aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
						qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
					else:
						aug_tensor_patch_list.append(None)
						qualified_tensor_patch_list.append(None)
			

			#get embeddings of qualified patches through student model
			model=model.train()
			model.module.contrast=True
			student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

			#get embeddings of augmented patches through teacher model
			teacher_model.train()
			teacher_model.contrast = True
			teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

			#enqueue these
			embd_queues.enqueue(teacher_embedding_list)

			#calculate PCGJCL loss
			PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 1 , 10, alpha=1)


			#calculate supervised loss 
			# labeled data 사용
			imgs2, masks2 =imgs2.to(dev), masks2.to(dev)
			out = model(imgs)
			supervised_loss = cross_entropy_loss(out,masks2)


			#Consistency Loss
			# unlabeled data 사용
			consistency_loss=consistency_cost(model,teacher_model,imgs,p_masks)
			# consistency_loss 구할 때 p_masks 가 필요없지 않나?


			#total loss
			loss = supervised_loss + 0.5*PCGJCL_loss + 4*consistency_loss
			
			#backpropagate
			loss.backward()
			optimizer_ssl.step()
			scheduler.step()


			for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
				param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

			#Extras
			t2=time.time()
			print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
			step+=1
		if epoch_loss < min_loss:
			torch.save(model,'./best_contrast.pth')

model = torch.load("C:/Users/herbw/OneDrive/#IIPL/2024_IEIE/PatchCL-MedSeg/best_model.pth")
model.eval()