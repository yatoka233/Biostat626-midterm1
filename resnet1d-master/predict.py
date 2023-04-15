import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from util import read_data_physionet_2, read_data_physionet_4, preprocess_physionet
from net1d import Net1D, MyDataset
from resnet1d import ResNet1D
from acnn1d import ACNN
from crnn1d import CRNN
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import focal loss

from tensorboardX import SummaryWriter
from torchsummary import summary
import argparse
from imblearn.over_sampling import SMOTE, ADASYN

train_root = '/home/dengfy/Midterm1/data/training_data.txt'
test_root = '/home/dengfy/Midterm1/data/test_data.txt'
def perturb(data_x):
    # perturb the data by adding random noise
    # data_x: data without label
    # return: perturbed data
    # add random noise
    noise = np.random.normal(0, 0.1, data_x.shape)
    # add noise to data_x
    data_x = data_x + noise
    return data_x


def get_offset(training_data, n=1):
    ## input training_data in original order
    offset_list = []
    offset_list2 = []
    for i in training_data['subject'].unique():
        subject = training_data[training_data['subject'] == i]
        ## [0:n-1] rows of training_data
        offset = subject[0:subject.shape[0]-1]
        # concate the first row of offset and offset 
        offset = pd.concat([offset[0:1], offset])
        # reset the index of offset same as subject
        idx = subject.index
        offset.index = idx
        # offset column names + .1
        offset.columns = [col + '.1' for col in offset.columns]
        offset_list.append(offset)

        if n==2:
            offset2 = subject[0:subject.shape[0]-2]
            offset2 = pd.concat([offset2[0:1], offset2[0:1], offset2])
            offset2.index = idx
            offset2.columns = [col + '.2' for col in offset2.columns]
            offset_list2.append(offset2)

    # concat offset_list
    offset = pd.concat(offset_list)
    if n==2:
        offset2 = pd.concat(offset_list2)
    # drop subject.1 activity.1
    if 'activity.1' in offset.columns:
        offset = offset.drop(['subject.1', 'activity.1'], axis=1)
    else:
        offset = offset.drop(['subject.1'], axis=1)

    if n==2:
        if 'activity.2' in offset2.columns:
            offset2 = offset2.drop(['subject.2', 'activity.2'], axis=1)
        else:
            offset2 = offset2.drop(['subject.2'], axis=1)
        offset = pd.concat([offset, offset2], axis=1)

    # concate training_data and offset
    s = pd.concat([training_data, offset], axis=1)
    
    return s

def get_data(root, task="task2", train=True, binary=True, offset=False):
    ## task = "task1" or "task2" or "test"


    ## read training_data.txt from second row
    training_data = pd.read_csv(root,sep='\t',skiprows=1, header=None)
    ## open training_data.txt and read the first row
    with open(root) as f:
        first_line = f.readline()
    ## split the first row by tab
    first_line = first_line.split()
    ## change the column name of training_data
    training_data.columns = first_line

    ## number of rows
    print("number of rows:", training_data.shape[0])

    if not train:
        if binary:
            pred = pd.read_csv('/home/dengfy/Midterm1/output/binary_task1.txt', header=None)
            # column name
            pred.columns = ['binary']
            training_data['binary'] = pred['binary']
        if offset:
            training_data = get_offset(training_data, n=offset)
        return training_data

    ## change the activity column
    if task == "task1":
        ## <=3 -> 1   others -> 0
        training_data['activity'] = training_data['activity'].apply(lambda x: 1 if x<=3 else 0)
    elif task == "task2":
        ## >=7 -> 7
        if binary:
            training_data['binary'] = training_data['activity'].apply(lambda x: 1 if x<=3 else 0)
        if offset:
            training_data = get_offset(training_data, n=offset)

        training_data['activity'] = training_data['activity'].apply(lambda x: 7 if x>=7 else x)



    return training_data

## seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, required=True)
    args = parser.parse_args()
    sid = args.sid
    seed = 42

    task = "task2"
    binary = True
    offset = 2

    is_debug = False
    batch_size = 32
    if is_debug:
        writer = SummaryWriter('/home/dengfy/resnet1d-master/debug')
    else:
        writer = SummaryWriter('/home/dengfy/resnet1d-master/layer98')

    # make data
    print("making data...")
    training_data = get_data(train_root, task=task, binary=binary, offset=offset)

    Y_train = training_data['activity'] ## y is activity
    Y_train = Y_train - 1
    # 3 -> 1, 4 -> 2, others -> 0
    Y_train = Y_train.replace(1, 0)
    Y_train = Y_train.replace(2, 0)
    Y_train = Y_train.replace(5, 0)
    Y_train = Y_train.replace(6, 0)
    Y_train = Y_train.replace(3, 1)
    Y_train = Y_train.replace(4, 2)

    X_train = training_data.drop(['activity','subject'], axis=1) ## x is all the other columns
    ## over sample
    oversample = ADASYN(sampling_strategy="not majority", random_state=42)
    X_, y_ = oversample.fit_resample(X_train, Y_train)
    X_train = X_
    Y_train = pd.DataFrame(y_)
    print(Y_train.value_counts())


    # X_train to tensor (sample, channel, length)
    X_train = torch.from_numpy(X_train.values).float()
    X_train = X_train.unsqueeze(1)
    Y_train = torch.from_numpy(Y_train.values.squeeze()).long()


    test_data = get_data(test_root, task=task, train=False, binary=binary, offset=offset)
    X_test = test_data.drop(['subject'], axis=1)
    X_test = torch.from_numpy(X_test.values).float()
    X_test = X_test.unsqueeze(1)
    # Y test = 0
    Y_test = torch.zeros(X_test.shape[0]).long()

    print(X_train.shape, Y_train.shape)

    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    
    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # model = Net1D(
    #     in_channels=1, 
    #     base_filters=256, 
    #     ratio=1.0, 
    #     filter_list=[64,160,160,400,400,1024,1024], 
    #     m_blocks_list=[2,2,2,3,3,4,4], 
    #     kernel_size=16, 
    #     stride=2, 
    #     groups_width=1,
    #     verbose=False, 
    #     n_classes=3)
    
    # model = ACNN(
    #     in_channels=1, 
    #     out_channels=128, 
    #     att_channels=16,
    #     n_len_seg=281, 
    #     verbose=False,
    #     n_classes=3,
    #     device='cuda')
    
    # model = CRNN(
    #     in_channels=1, 
    #     out_channels=16, 
    #     n_len_seg=281, 
    #     verbose=False,
    #     n_classes=3,
    #     device='cuda')
    
    ## change the hyper-parameters for your own data
    # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
    # 34 layer (16*2+2): 16, 2, 4
    # 98 layer (48*2+2): 48, 6, 12
    model = ResNet1D(
        in_channels=1, 
        base_filters=128, 
        kernel_size=16, 
        stride=2, 
        n_block=48, 
        groups=4,
        n_classes=3, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        verbose=False)

    
    model.to(device)

    # summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    ## add weights to 3 and 4
    # weights = torch.tensor([1,5,1]).float()
    # loss_func = nn.CrossEntropyLoss(weight=weights.to(device))

    # loss_func = nn.CrossEntropyLoss()

    ## focal loss
    focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='focal_loss',
        alpha=[0.2, 1.0, 1.2], # [.2, 1.0, 1.2]
        gamma=2,
        reduction='mean',
        device='cuda',
        dtype=torch.float32,
        force_reload=False
    )
    loss_func = focal_loss

    n_epoch = 20
    step = 0
    best_acc = 0
    best_loss = 100
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        ## accumulate acc
        all_pred = []
        all_true = []
        loss_list = []
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            # print(pred.shape, input_y.shape)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            loss_list.append(loss.item())

            pred = torch.argmax(pred, dim=1)
            all_pred.append(pred.cpu().data.numpy())
            all_true.append(input_y.cpu().data.numpy())
            accumuated_pred = np.concatenate(all_pred)
            accumuated_true = np.concatenate(all_true)
            acc = accuracy_score(accumuated_true, accumuated_pred)
            prog_iter.set_description("Training (loss=%.4f) (acc=%.4f)" % (loss.item(), acc))

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break
        
        
        scheduler.step(_)
                    
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred_test = np.argmax(all_pred_prob, axis=1)


        # softmax
        all_pred_prob = np.exp(all_pred_prob)
        all_pred_prob = all_pred_prob / np.sum(all_pred_prob, axis=1, keepdims=True)
        max_pred_prob = np.max(all_pred_prob, axis=1)

        # save  prediction to txt
        loss_mean = np.mean(loss_list)
        np.savetxt('pred45/pred45_' + str(_) + '.txt', all_pred_test, fmt='%d')
        np.savetxt('pred45/pred_prob45_' + str(_) + '.txt', max_pred_prob, fmt='%.4f')
            



    




    