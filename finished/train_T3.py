import os
import pickle
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter
from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import MultiDimAverageMeter, EMA
import random

seed =22
random_seed = seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def calculate_cosine_similarity_loss(model_b, model_d, layer_names):
    p11 = list(model_b.parameters())
    print(f'p11: {p11}')
    print(f'p11[8]:{p11[8]}')
    p22 = list(model_d.parameters())
    cos_sim = 0.0
    for i in layer_names:
            p111 = p11[i].view(p11[i].size(0), -1)
            p222 = p22[i].view(p22[i].size(0), -1)
            sim = F.cosine_similarity(p111, p222)
            cos_sim += sim.mean(0)
            # print(f'cos_sim: {cos_sim}')

    return cos_sim


@ex.automain
def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_optimizer_tag,
    main_learning_rate,
    main_weight_decay,
    lambda_ortho = 0.01,
):
    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))
    train_dataset = get_dataset(dataset_tag, data_dir=data_dir, dataset_split="train", transform_split="train")
    valid_dataset = get_dataset(dataset_tag, data_dir=data_dir, dataset_split="eval", transform_split="eval")
    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = [torch.max(train_target_attr).item() + 1, torch.max(train_bias_attr).item() + 1]
    num_classes = attr_dims[0]
    train_dataset, valid_dataset = IdxDataset(train_dataset), IdxDataset(valid_dataset)
    train_loader = DataLoader(train_dataset, batch_size=main_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
    model_b, model_d = get_model(model_tag, attr_dims[0]).to(device), get_model(model_tag, attr_dims[0]).to(device)
    
    if main_optimizer_tag == "SGD":
        optimizer_b = torch.optim.SGD(model_b.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay, momentum=0.9)
        optimizer_d = torch.optim.SGD(model_d.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay, momentum=0.9)
    elif main_optimizer_tag == "Adam":
        optimizer_b = torch.optim.Adam(model_b.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay)
    elif main_optimizer_tag == "AdamW":
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay)
        optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay)
    else:
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss()
    sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
    sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

    def evaluate(model, data_loader):
        model.eval()
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data, attr, label = data.to(device), attr.to(device), label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
            attr = attr[:, [target_attr_idx, bias_attr_idx]]
            attrwise_acc_meter.add(correct.cpu(), attr.cpu())
        accs = attrwise_acc_meter.get_mean()
        model.train()
        return accs

    valid_attrwise_accs_list = []
    num_updated = 0
    
    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)
        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]
        logit_b = model_b(data)
        if np.isnan(logit_b.mean().item()):
            print(logit_b)
            raise NameError('logit_b')
        logit_d = model_d(data)
        loss_b = criterion(logit_b, label).cpu().detach()
        loss_d = criterion(logit_d, label).cpu().detach()
        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d')
        
        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d
        
        # EMA sample loss
        sample_loss_ema_b.update(loss_b, index)
        sample_loss_ema_d.update(loss_d, index)
        
        # class-wise normalize
        loss_b = sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = sample_loss_ema_d.parameter[index].clone().detach()
        
        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b_ema')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d_ema')
        
        label_cpu = label.cpu()
        
        for c in range(num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d
            
        # re-weighting based on loss value / generalized CE for biased model
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')
            
        loss_b_update = bias_criterion(logit_b, label)

        if np.isnan(loss_b_update.mean().item()):
            raise NameError('loss_b_update')
        loss_d_update = criterion(logit_d, label) * loss_weight.to(device)
        if np.isnan(loss_d_update.mean().item()):
            raise NameError('loss_d_update')
        

#--------------------------------------------------------------------------------------------#
        layer_names = [2,4]
        ortho_loss = calculate_cosine_similarity_loss(model_d, model_b, layer_names)
        loss = loss_b_update.mean() + loss_d_update.mean() - (lambda_ortho / 2) * ortho_loss
#--------------------------------------------------------------------------------------------#

        num_updated += loss_weight.mean().item() * data.size(0)
        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()
        







        if step % main_valid_freq == 0:
            valid_attrwise_accs_b = evaluate(model_b, valid_loader)
            valid_attrwise_accs_d = evaluate(model_d, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs_d)
            valid_accs_b = torch.mean(valid_attrwise_accs_b)
            writer.add_scalar("acc/b_valid", valid_accs_b, step)
            valid_accs_d = torch.mean(valid_attrwise_accs_d)
            writer.add_scalar("acc/d_valid", valid_accs_d, step)
            eye_tsr = torch.eye(attr_dims[0]).long()
            eye_tsr = torch.eye(attr_dims[0]).long()
            writer.add_scalar("acc/b_valid_aligned", valid_attrwise_accs_b[eye_tsr == 1].mean(), step)
            writer.add_scalar("acc/b_valid_skewed", valid_attrwise_accs_b[eye_tsr == 0].mean(), step)
            writer.add_scalar("acc/d_valid_aligned", valid_attrwise_accs_d[eye_tsr == 1].mean(), step)
            writer.add_scalar("acc/d_valid_skewed", valid_attrwise_accs_d[eye_tsr == 0].mean(), step)
            num_updated_avg = num_updated / main_batch_size / main_valid_freq
            writer.add_scalar("num_updated/all", num_updated_avg, step)
            num_updated = 0
    
    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    model_b_file_name = f"model_b_cosine__neg__{lambda_ortho}__t3.th"
    model_d_file_name = f"model_d_cosine__neg__{lambda_ortho}__t3.th"
    model_b_path = os.path.join(log_dir, "result", main_tag, model_b_file_name)
    model_d_path = os.path.join(log_dir, "result", main_tag, model_d_file_name)

    state_dict_b = {'steps': step, 'state_dict': model_b.state_dict(), 'optimizer': optimizer_b.state_dict()}
    with open(model_b_path, "wb") as f:
        torch.save(state_dict_b, f)
    state_dict_d = {'steps': step, 'state_dict': model_d.state_dict(), 'optimizer': optimizer_d.state_dict()}
    with open(model_d_path, "wb") as f:
        torch.save(state_dict_d, f)