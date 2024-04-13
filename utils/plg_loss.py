import torch 
import numpy as np
from hard_neg_mining import hard_neg

def PCGJCL(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
  
    """
    INPUT :
    patch_list = list of length num_class_in_patches containing tensors t_i
    # list len 4 인 teacher_embedding_list
    t_i = tensor of shape num_samples_class_i*dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT :
    loss = tensor
    """
    num_classes= len(patch_list) # =4개
    total_samples=0
    mean_list=[]
    cov_list=[]
    loss = torch.tensor([0])
    num_classes_in_batch=0
    #calculate mean and cov matrices for each class 

    # class=1
    print(f"patch_list[3].shape {patch_list[3].shape}") # torch.size([4,128])


    for i in range(num_classes):
        if patch_list[i] is not None: 
            t_i = torch.stack(embd_queues.queues[i], dim=0) #128이어야함
            # 지금 embd_queues.queues[i] 이게 하나의 class에 해당하는 임베딩 (32,64,64)

            # mbd_queues.queues[i]
            # t_i = t_i.view(32,-1) ## 내가 추가:  flatten
            print(f't_i.shape(cat):{t_i.shape}')
            total_samples += patch_list[i].shape[0]
            num_classes_in_batch+=1
            mu=torch.mean(t_i, dim=0) # 4 x 128
            mean_list.append(mu)
            sig =torch.mm((t_i-mu).t(), (t_i-mu))/(t_i.shape[0])
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    g_count=0
    den_lists=[]
    pos_lists=[]
    for i in range(num_classes):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i=patch_list[i], mean_list[i], cov_list[i]
            print(t_i.shape) # 288,128
            print(mu_i.shape) # 16384
            num = (-torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0))/tau
            #den_neg, den_pos = torch.tensor([0]), torch.tensor([0])
            l_count=0
            #Iterate over neg classes for a particluar class
            for j in range(num_classes):
                if patch_list[j] is not None:
                    if i!=j:
                        t_j, mu_j, sig_j= torch.stack(embd_queues.queues[j], dim=0), mean_list[j], cov_list[j]
                        #get hard neg mu_j,sig_j
                        hn_mu_j, hn_sig_j = hard_neg(t_i, mu_i, sig_i, t_j, mu_j, sig_j, 0.5)# GET HARD NEGS HERE
                        #hn_mu_j, hn_sig_j = mu_j, sig_j # USE THIS FOR NO HARD NEGS
                        if l_count==0:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            l_count+=1
                            den_lists.append([den_neg])
                        else:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            den_lists[i].append(den_neg)
                
                    else:
                        den_pos = torch.mm(t_i, mu_i.view(emb_dim, 1))/tau + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(sig_i, t_i.t())))).view(-1,1))
                        pos_lists.append(den_pos)
                # else:
                #     den_lists[i].append(None)
        
            a=pos_lists[i]
            res=torch.zeros(a.size())
            res+=torch.exp(a)       
            for d in den_lists[i]:
              res+=(psi/num_classes_in_batch)*torch.exp(d)# ADDED WEIGHT FOR NEG TERMS HERE ....psi*den_neg+den_pos
            den = torch.sum(torch.log(res), dim=0)
            if g_count==0:
                loss= num+den
                g_count+=1
            else:
                loss+=num+den
        
        else:
          pos_lists.append(None)
          den_lists.append(None)

    loss = loss/total_samples
    
    return loss
