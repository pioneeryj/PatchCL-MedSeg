import math
import cv2
import torch
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import sys 
sys.path.append("C:/Users/herbw/OneDrive/#IIPL/2024_IEIE/PatchCL-MedSeg/utils")
from nms_utils import non_max_suppression_slow, non_max_suppression_fast
import matplotlib.pyplot as plt

def patcher(imgs,mask,x,y,w,h,patch_size,image_size):

    """
    INPUTS:
    imgs(ndarray): image of shape 3 X image_size X image_shape
    mask(ndarray): binary mask of image_size X image_size
    x(int): upper left x coordinate
    y(int): upper left y coordinate
    w(int): contour width
    h(int): contour length
    patch_size(int)
    image_size(int)

    OUPUTS:
    out: list of patches wich patch shape as (3 X patch_shape X patch_shape)
    """
    
    if w<patch_size and h<patch_size:
        #flag1,flag2 indicates if patch centered at x,y exceeds the image x,y dims from upper left portions
        #flag3,flag4 ndicates if patch centered at x,y exceeds the image x,y dims from lower right postions
        flag1 = 1 if x+w//2-patch_size//2<0 else 0
        flag2 = 1 if y+h//2-patch_size//2<0 else 0
        flag3 = 1 if x+w//2+patch_size//2>image_size else 0
        flag4 = 1 if y+h//2+patch_size//2>image_size else 0
        
        #note that flag1 & flag3 cant be 1 simultaneously. Similarly with flag2 and flag4
        sum=flag1+flag2+flag3+flag4
        if sum is 0:
            x_corner, y_corner = x+w//2-patch_size//2, y+h//2-patch_size//2
            return [imgs[:,x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]],[np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size])]
        elif sum is 1:
            x_corner, y_corner = (x+w//2-patch_size//2)*(1-flag1)*(1-flag3)+flag3*(image_size-patch_size), (y+h//2-patch_size//2)*(1-flag2)*(1-flag4)+flag4*(image_size-patch_size)
            return [imgs[:,x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]],[np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size])]
        else:
            x_corner, y_corner = (1-flag1)*(1-flag3)+flag3*(1-flag1)*(image_size-patch_size),(1-flag1)*(1-flag3)+flag3*(1-flag1)*(image_size-patch_size)
            #flag1,flag2=1 OR flag3,flag4=1
            return [imgs[:,x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size]],[np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size])]
    else:
        #flag1,flag2 indicates if patch centered at x,y exceeds the image x,y dims from upper left portions
        #flag3,flag4 ndicates if patch centered at x,y exceeds the image x,y dims from lower right postions
        flag1 = 1 if x+w//2-patch_size//2<0 else 0
        flag2 = 1 if y+h//2-patch_size//2<0 else 0
        flag3 = 1 if x+w//2+patch_size//2>image_size else 0
        flag4 = 1 if y+h//2+patch_size//2>image_size else 0

        sum=flag1+flag2+flag3+flag4 #sum cannot be 2 here
        if h<patch_size or w<patch_size:
            if sum is 0:
                x = x+w//2-patch_size//2 if w<patch_size else x
                y = y+h//2-patch_size//2 if h<patch_size else y
            elif sum is 1: #change x OR y for suitable cropping
                x = 0 + (image_size-patch_size)*flag3*(1-flag1)+x*(1-flag1)*(1-flag3)
                y = 0 + (image_size-patch_size)*flag4*(1-flag2)+y*(1-flag2)*(1-flag4)
        
        out_list=[]
        coordinate_list=[]
        horizontal_components= int(math.ceil(w/patch_size))
        vertical_components= int(math.ceil(h/patch_size))
        for i in range(horizontal_components):
            for j in range(vertical_components):
                if i != horizontal_components-1 and j != vertical_components-1:
                    x_corner, y_corner = x+i*patch_size,y+j*patch_size
                    out_list.append(imgs[:,x_corner:x_corner+patch_size, y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size, y_corner:y_corner+patch_size])
                    coordinate_list.append(np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size]))
                elif i == horizontal_components-1 and j != vertical_components-1:
                    x_corner, y_corner = x+w-patch_size if w>=patch_size else x, y+j*patch_size
                    out_list.append(imgs[:,x_corner:x_corner+patch_size, y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size])
                    coordinate_list.append(np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size]))
                elif i != horizontal_components-1 and j == vertical_components-1:
                    x_corner, y_corner = x+i*patch_size, y+h-patch_size if h>=patch_size else y
                    out_list.append(imgs[:,x_corner:x_corner+patch_size, y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size])
                    coordinate_list.append(np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size]))
                else:
                    x_corner, y_corner = x+w-patch_size if w>=patch_size else x, y+h-patch_size if h>=patch_size else y
                    out_list.append(imgs[:,x_corner:x_corner+patch_size, y_corner:y_corner+patch_size]*mask[x_corner:x_corner+patch_size,y_corner:y_corner+patch_size])
                    coordinate_list.append(np.array([x_corner, y_corner,x_corner+patch_size, y_corner+patch_size]))
        return out_list,coordinate_list

def convert_to_one_hot(masks, num_classes=4):
    """
    마스크 텐서를 원-핫 인코딩 형식으로 변환합니다.
    """
    batch_size, height, width = masks.shape
    one_hot = np.zeros((batch_size, num_classes, height, width), dtype=np.uint8)
    for class_id in range(num_classes):
        class_mask = (masks == class_id)
        one_hot[:, class_id, :, :] = class_mask # 0 또는 1

    return one_hot

def pad_patch(patch, patch_size):
    """
    Ensure the patch is exactly patch_size x patch_size.
    """
    current_height, current_width = patch.shape[-2:]
    pad_height = max(0, patch_size - current_height)
    pad_width = max(0, patch_size - current_width)
    patch = np.pad(patch, ((0,0), (pad_height//2, pad_height - pad_height//2), (pad_width//2, pad_width - pad_width//2)), mode='constant', constant_values=0)
    return patch

# 패치 사이즈 지정
def _get_patches(imgs,masks,classes=4,background=True,img_size=384,patch_size=64):

    """
    INPUTS:
    'classes' (int): the number of classes in the dataset
    'background' (Boolean): is background considered a class in the dataset
    'img_size' (int)= size of input images
    'patch_size' (int)= size of patches
    'masks' (tensor): input masks of shape (batch X 'classes' X 'img_size' X 'img_size')
    'imgs' (tesnor): input images of shape (batch X channels X img_size X img_size)

    OUTPUTS:
    'patches' (list of ndarrays): list of len 'classes' made of ndarrays of shape (NumberOfPatches X 3 X 'patch_size' X 'patch_size')
    """

    #LOOK AT THIS CONVERSTION!!
    masks=np.array(masks,dtype='uint8')
    one_hot_masks = convert_to_one_hot(masks, num_classes=4)
    imgs=np.array(imgs)
    print(f'비포 imgs.shape{imgs.shape}')
    imgs = np.repeat(imgs[..., np.newaxis], 3, axis=-1)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    #imgs=np.expand_dims(imgs, axis=1)# 채널 차원 추가해줬음.
    print(f'애프터 imgs.shape{imgs.shape}')

    #print(f"masks shape:{masks.shape}") #(8, 384, 276)

    patches=[]
    start_index=0


    #Assuming background index to be 0
    if background is True:
        start_index+=1
        # bkgrnds=masks[:,0,:,:] #getting the background masks
        bkgrnds=one_hot_masks[:,0,:,:] #getting the background masks
        bkgrnds_list=[]
        num_bkgrnd_patches=int(math.ceil(img_size/patch_size))
        for i in range(num_bkgrnd_patches):
            for j in range(num_bkgrnd_patches):
                patch_list = []
                for k in range(bkgrnds.shape[0]):
                    if i != num_bkgrnd_patches-1 and j != num_bkgrnd_patches-1:
                        patch = imgs[k, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] * bkgrnds[k, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    elif i == num_bkgrnd_patches-1 and j != num_bkgrnd_patches-1:
                        patch = imgs[k, :, img_size-patch_size:img_size, j*patch_size:(j+1)*patch_size] * bkgrnds[k, img_size-patch_size:img_size, j*patch_size:(j+1)*patch_size]
                    elif i != num_bkgrnd_patches-1 and j == num_bkgrnd_patches-1:
                        patch = imgs[k, :, i*patch_size:(i+1)*patch_size, img_size-patch_size:img_size] * bkgrnds[k, i*patch_size:(i+1)*patch_size, img_size-patch_size:img_size]
                    else:
                        patch = imgs[k, :, img_size-patch_size:img_size, img_size-patch_size:img_size] * bkgrnds[k, img_size-patch_size:img_size, img_size-patch_size:img_size]
                    patch_list.append(pad_patch(patch, patch_size))

                bkgrnds_list.extend(patch_list)
        patches.append(np.stack(bkgrnds_list, axis=0) if len(bkgrnds_list) > 0 else None)
    
    for cls_index in range(start_index,classes):
        masks_=one_hot_masks[:,cls_index,:,:] #getting the class masks [:,:,:]
        cls_list=[]
        cls_coordinate=[]
        for im_index in range(masks_.shape[0]):
            if np.sum(masks_[im_index,:,:])>0:
                contours,heirar = cv2.findContours(masks_[im_index,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for conts in contours:
                    y,x,h,w=cv2.boundingRect(conts)
                    #plt.imshow(cv2.rectangle(masks_[im_index,:,:],(y,x),(y+h,x+w),color=(1),thickness=3))
                    #plt.imshow(cv2.UMat.get(cv2.rectangle(imgs[im_index,:,:,:].transpose(1,2,0),(y,x),(y+h,x+w),color=(0,1,0),thickness=3)))
                    # plt.axis('off')
                    # plt.show()
                    if h*w>10:
                        patches_from_countour, patch_coordinates_from_contour = patcher(imgs[im_index,:,:,:],masks_[im_index,:,:],x,y,w,h,patch_size,img_size)
                        cls_list=cls_list+patches_from_countour
                        cls_coordinate=cls_coordinate+patch_coordinates_from_contour

        if len(cls_list)>0:
            all_cls_patches = np.stack(cls_list,axis=0)
            all_cls_coordinates = np.stack(cls_coordinate,axis=0)
            _,remove_indices = non_max_suppression_slow(all_cls_coordinates,overlapThresh=0.8) #overlapThresh = 1 means no overlap at all; 0 means total overlap
            all_cls_patches = np.delete(all_cls_patches,remove_indices,axis=0)
            patches.append(all_cls_patches)
        else:
            patches.append(None)
    
    print(f'## patches[3].shape: {patches[3].shape}')

    def visualize_channel(data, channel_index=0):
        if data.ndim != 4 or data.shape[1] < channel_index + 1:
            raise ValueError("데이터는 (N, C, H, W) 형태이어야 하며, 주어진 채널 인덱스가 유효해야 합니다.")

        # 데이터에서 선택한 채널 이미지를 가져옵니다.
        # data.shape 예시: (1, 3, 192, 192), channel_index 예시: 0 (첫 번째 채널)
        image = data[0, channel_index, :, :]  # 첫 번째 이미지의 특정 채널

        # 이미지 시각화
        plt.imshow(image, cmap='gray')
        plt.colorbar()  # 색상 막대 표시
        plt.title(f"Channel {channel_index}")
        plt.show()

    data=patches[3]
    visualize_channel(data, channel_index=0)
    return patches
                    
