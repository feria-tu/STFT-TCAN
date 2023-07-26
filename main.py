'''
将频域和时域合并输入到模型
version：短时傅里叶变换stft，只能时域频域分开输入模型
'''
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from time import time
from tqdm import tqdm
from data_convert import convert_to_freq
from models.tcan import WTTCAN
from utils.pot import pot_eval
from utils.spot import SPOT
from utils.parser import *
from utils.dlutils import *
import torch.optim as optim
from models.models import *
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能由于是MacOS系统的原因会出现报错

# global constant
datasets = ['SMAP', 'MSL', 'SMD', 'MBA', 'WADI', 'SWaT']
# data folder
output_folder = 'processed'
data_folder = 'datasets'
freq_folder = 'frequence'

# 取20%数据集的数据，验证模型在少量数据集上的表现情况
def cut_val(percentage, arr):
    print('Slicing dataset to 20%')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]

# 划分为滑动时间窗口
def convert_to_window(data,model):
    windows = []
    w_size = model.n_window
    for i,g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i] # 多切
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]]) # 少补
        # windows.append(w)
        windows.append(w if 'DTAAD' in args.model or 'STFT' in args.model or 'TranAD' in args.model  else w.view(-1))
    return torch.stack(windows) # 在增加新的维度进行堆叠

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname,dims):
    import models.models
    model_class = getattr(models.models,modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.test):
        print(f"Loading pre-trained model: {model.name}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"Creating new model: {model.name}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def load_dataset(dataset):
    folder = os.path.join(output_folder,dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    train_freq,test_freq = load_freq(dataset)
    train_freq,test_freq = train_freq.astype(np.double),test_freq.astype(np.double)
    if not os.path.exists(folder):
        raise Exception('Data Not Found.')
    for file in ['train','test','labels']:
        if dataset == 'SMD':
            file = 'machine-1-1_' + file
        if dataset == 'SMAP':
            file = 'P-1_' + file
        if dataset == 'MSL':
            file = 'C-1_' + file
        loader.append(np.load(os.path.join(folder,f'{file}.npy')))
        # loader包括loader[0]：train，loader[1]：test，loader[2]：label
    
    # 以NAB数据集为例：loader shape=(3, 4032, 1)
    # DataLoader 是一个迭代器，最基本的使用方法就是传入一个 Dataset 对象，它会根据参数 batch_size 的值生成一个 batch 的数据

    # 是否需要转换成频域数据
    if args.freq:     
        loader_train = np.concatenate((loader[0],train_freq),axis=0)
        loader_test = np.concatenate((loader[1],test_freq),axis=0)
        # 时频结合
        train_loader = DataLoader(loader_train, batch_size=loader[0].shape[0])
        test_loader = DataLoader(loader_test, batch_size=loader[1].shape[0])
    else:
        # 仅时域数据
        train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])

    # 检测在少量数据集上的效果，这里取20%实验集
    if args.twenty: 
        loader_train = cut_val(0.2, loader_train)
        loader[0] = cut_val(0.2, loader[0])

    labels = loader[2]
    return train_loader, test_loader, labels

def load_freq(dataset):
    folder = os.path.join(freq_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Frequnce Data not found.')
    # print(folder)
    for file in ['train', 'test']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'MBA': file = file
        path = os.path.join(folder, f'{file}_freq.npy')
        freq = np.load(path,allow_pickle=True)
        if 'train' in file:
            train_freq = freq
        if 'test' in file:
            test_freq = freq
    return train_freq,test_freq

def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            model.to(torch.device(args.Device))
            for i, d in enumerate(data):
                d = d.to(torch.device(args.Device))
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                # loss = loss.sum()
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                # loss.backward(gradient=torch.tensor(loss))
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data: 
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['MTAD_GAT']:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                d = d.to(torch.device(args.Device))
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            xs = []
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'MAD_GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                d = d.to(torch.device(args.Device))
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size = bs)
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'DTAAD' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2),elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            for d, _ in dataloader:
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, bs, feats)
                z = model(window)
                z = z[1].permute(1, 0, 2)
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'STFT' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        data_x = torch.DoubleTensor(data)
        # data_x = torch.Tensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = 10
        l1s, l2s = [], []
        if training:
            for d,_ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2),elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            for d, _ in dataloader:
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, bs, feats)
                z = model(window)
                z = z[1].permute(1, 0, 2)
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    else:
        model.to(torch.device(args.Device))
        data = data.to(torch.device(args.Device))
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            return loss.detach().numpy(), y_pred.detach().numpy()
    

if __name__ == '__main__':
    # 先进行数据预处理
    # 加载数据集
    print('Usage: python main.py --dataset <dataset> --model <model> [--freq <bool>] [--twenty]')
    print('dataset: SMAP/MSL/SMD/MBA/WADI/SWaT')
    if args.dataset not in ['SMAP','MSL','SMD','MBA','WADI','SWaT']:
        raise Exception(f'Unknown Dataset ',args.dataset)
    # 将时域数据转为频域数据
    if not os.path.exists(freq_folder):
        convert_to_freq()

    train_loader,test_loader,labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    # # prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['USAD', 'MTAD_GAT',
                      'MAD_GAN', 'TranAD', 'DTAAD'] or 'STFT' in model.name: 
        trainD, testD = convert_to_window(trainD,model),convert_to_window(testD,model)

    # # 将数据输入TCAN 模型
    # # 初始化模型和优化器
    # input_dim = 10  # 输入数据的特征维度
    # output_dim = 1  # 输出大小
    # kernel_size = 3  # 卷积核大小
    # dropout = 0.2  # dropout率

    # Hyperparameters lr，根据不同的数据集定不同的lr
    lr_d = {
        'SMD': 0.0001,
        'SMAP': 0.001,
        'MBA': 0.001,
        'SWaT': 0.009,
        'WADI': 0.0001,
        'MSL': 0.002,
    }

    print(f'======================Training model on {args.dataset}======================')

    # model = WTTCAN(feats=labels.shape[1],lr=lr_d).double()
    # optimizer = optim.AdamW(model.parameters(),lr=lr_d[args.dataset], weight_decay=1e-5)
    # criterion = nn.MSELoss()  # 二元交叉熵损失函数
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    
    # training ***********************************
    start = time()
    if not args.test:
        num_epochs = 5
        e = epoch + 1
        batch_size = 128
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(
                e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        save_model(model, optimizer, scheduler, e, accuracy_list)
        print('Training time: ' + "{:10.4f}".format(time() - start) + ' s' )

    # testing ***********************************
    start = time()
    torch.zero_grad = True
    model.eval()
    print(f'======================Testing model on {args.dataset}======================')
    loss,y_pred = backprop(
        0,model,testD,testO,optimizer,scheduler,training=False)
    print('Testing time: ' + "{:10.4f}".format(time() - start) + ' s' )
    
    # Score ***********************************
    df = pd.DataFrame()
    lossT,_ = backprop(
        0,model,trainD,trainO,optimizer,scheduler,training=False)

    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        # POT: 使用峰值过阈值（POT）方法自动选择异常阈值
        result, pred = pot_eval(lt, l, ls)
        df = df.append(result, ignore_index=True)
    
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    print(df)
    pprint(result)




        

 

