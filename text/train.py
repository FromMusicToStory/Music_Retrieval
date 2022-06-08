import argparse
import random
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from model import *
from dataloader import *


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--is_training',
        default=True,
        required=False,
        help='run train'
    )
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        required=False,
        help='epochs'
    )
    parser.add_argument(
        '--batch',
        default=16,
        type=int,
        required=False,
        help='batch size'
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        required=False,
        help='shuffle'
    )
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        required=False,
        help='learning rate'
    )

    parser.add_argument(
        '--cuda',
        default='cuda:0',
        help='class weight'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='save checkpoint'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='test',
        help='checkpoint name to load or save'
    )

    parser.add_argument(
        '--hidden',
        action='store_true'
    )
    args = parser.parse_args()
    return args

args = parse_args()



def train(model,optimizer, dataloader):
    print("Train start")
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1).to(args.cuda)
    tqdm_train = tqdm(total=len(dataloader), position=1)

    for batch_id, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(args.cuda)
        mask = batch['mask'].to(args.cuda)
        targets = batch['targets'].to(args.cuda)

        outputs = model(input_ids=input_ids,
                        attention_mask=mask,
                        do_clf=True)
        loss = loss_func(outputs.to(args.cuda), targets.to(args.cuda))

        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()
        loss = loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tqdm_train.close()


def evaluate(model,valid_dataset):
    print("Eval start")
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        dataloader = DataLoader(valid_dataset, args.batch)
        losses = 0
        pred = []
        labels = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.cuda)
            mask = batch['mask'].to(args.cuda)
            targets = batch['targets'].to(args.cuda)

            outputs = model(input_ids = input_ids,
                            attention_mask = mask,
                            do_clf=True)
            loss = loss_func(outputs.to(args.cuda), targets.to(args.cuda))
            outputs = outputs.max(dim=1)[1].tolist()

            loss = loss.item()
            losses += loss

            label = targets.tolist()
            labels.extend(label)
            pred.extend(outputs)

        losses = losses/len(valid_dataset)
        acc = accuracy_score(labels,pred) * 100
        recall = recall_score(labels,pred,average='weighted') * 100
        precision = precision_score(labels,pred,average='weighted') * 100
        f1 = f1_score(labels,pred,average='weighted') * 100
        confusion = confusion_matrix(labels,pred)

        print('Validation Result: Loss - {:.5f} | Acc - {:.3f} |\
 Recall - {:.3f} | Precision = {:.3f} | F1 - {:.3f}'.format(losses,acc,recall,precision,f1))

    return losses, acc, recall, precision, f1, confusion


def main():
    TEXT_DIR = '../dataset/Story_dataset/'
    if args.is_training == True:
        dataset = TextClassification_Dataset(TEXT_DIR, split='train')
        valid_data = TextClassification_Dataset(TEXT_DIR, split='valid')

        seed = 1024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = TextEncoder()
        device = args.cuda
        print('---------------------',device)
        model = model.to(device)
        print('---config---')
        print(args)
        max_f1 = float("-inf")
        min_loss = float("inf")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)


        for epoch in range(args.epochs):
            dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=args.shuffle)
            train(model, optimizer, dataloader)
            loss, acc, recall, precision, f1, confusion = evaluate(model,valid_data)
            print('-'*10,'confusion matrix of epoch {}'.format(epoch+1),'-'*10)
            print(confusion)

            if min_loss > loss:
                temp = min_loss
                min_loss = loss
                if 'ckpt' not in os.listdir():
                    os.mkdir('ckpt')
                if args.save and epoch > 20:
                    torch.save(model,'./ckpt/{}_epoch{}.pt'.format(args.model_name,epoch))
                    print("-"*10,"Save Model loss {:.4f} ->  {:.4f}".format(temp,min_loss),"-"*10)


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
