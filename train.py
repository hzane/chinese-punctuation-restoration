import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import random

import torch
from torch import nn, optim
from transformers import BertTokenizer

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from model import     BertPunc
from data_1_to_1 import load_file, preprocess_data, create_data_loader

warnings.filterwarnings("ignore", category = UndefinedMetricWarning)


def validate(
    model,
    criterion,
    epoch,
    epochs,
    iteration,
    iterations,
    data_loader_valid,
    save_path,
    train_loss,
    best_val_loss,
    best_model_path,
    punctuation_enc,
    best_f1_sum,
):

    val_losses = []
    val_accs = []
    val_f1s = []

    label_vals = list(punctuation_enc.values())
    print('data_len', len(data_loader_valid))
    for inputs, labels in tqdm(data_loader_valid, total = len(data_loader_valid)):

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()

            # NOTE output: batch_first [B*S, num_class]
            output = model(inputs)
            # label: [B, S] -> [B*S, ]
            labels = labels.view(-1)

            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.argmax(dim = 1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(
                metrics.f1_score(
                    y_true,
                    y_pred,
                    average = None,
                    labels = label_vals,
                )
            )

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis = 0)

    improved = ''

    if val_loss < best_val_loss:
        improved = '*'
        model_path = save_path + 'model'
        print("save epoch:{} iter:{} model".format(epoch + 1, iteration))
        torch.save(model.state_dict(), model_path)
        best_val_loss = val_loss
        best_model_path = model_path

    f1_vals = ';'.join(['{:.4f}'.format(val) for val in val_f1])

    print(
        "Epoch: {}/{}".format(epoch + 1, epochs),
        "Iteration: {}/{}".format(
            iteration,
            iterations,
        ),
        "Loss: {:.4f}".format(train_loss),
        "Val Loss: {:.4f}".format(val_loss),
        "Acc: {:.4f}".format(val_acc),
        "F1: {}".format(f1_vals),
        improved,
    )

    return best_val_loss, best_model_path, best_f1_sum


def train(
    model,
    optimizer,
    criterion,
    epochs,
    data_loader_train,
    data_loader_valid,
    save_path,
    punctuation_enc,
    iterations = 3,
    best_val_loss = 1e9
):

    print_every = len(data_loader_train) // iterations + 1
    best_model_path = None
    model.train()
    best_f1_sum = 0
    # print('data_len', print_every)
    pbar = tqdm(total = print_every)

    # print('epoch size', epochs)
    for e in range(epochs):

        counter = 1
        iteration = 1

        for inputs, labels in data_loader_train:
            inputs, labels = inputs.cuda(), labels.cuda()
            # print('print_first shape:', inputs.shape)

            inputs.requires_grad = False
            labels.requires_grad = False
            # NOTE output: batch_first [B*S, num_class]
            output = model(inputs)
            # label: [B, S] -> [B*S, ]
            labels = labels.view(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path, best_f1_sum = validate(
                    model,
                    criterion,
                    e,
                    epochs,
                    iteration,
                    iterations,
                    data_loader_valid,
                    save_path,
                    train_loss,
                    best_val_loss,
                    best_model_path,
                    punctuation_enc,
                    best_f1_sum,
                )
                model.train()
                pbar = tqdm(total = print_every)
                iteration += 1

            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path, best_f1_sum = validate(
            model,
            criterion,
            e,
            epochs,
            iteration,
            iterations,
            data_loader_valid,
            save_path,
            train_loss,
            best_val_loss,
            best_model_path,
            punctuation_enc,
            best_f1_sum,
        )
        model.train()
        if e < epochs - 1:
            pbar = tqdm(total = print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss


if __name__ == '__main__':

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(20)

    # 中文数据集标点符号
    punctuation_enc = {
        'O': 0,
        '，': 1,
        '。': 2,
        '？': 3,
    }

    segment_size = 100
    dropout = 0.3
    epochs_top = 10
    iterations_top = 3

    batch_size_top = 40
    learning_rate_top = 1e-5
    epochs_all = 15
    iterations_all = 3
    batch_size_all = 10
    learning_rate_all = 1e-5
    # 一句话的长度
    seq_len = 200
    hyperparameters = {
        'segment_size': segment_size,
        'dropout': dropout,
        'epochs_top': epochs_top,
        'iterations_top': iterations_top,
        'batch_size_top': batch_size_top,
        'learning_rate_top': learning_rate_top,
        'epochs_all': epochs_all,
        'iterations_all': iterations_all,
        'batch_size_all': batch_size_all,
        'learning_rate_all': learning_rate_all,
        'seq_len': seq_len,
    }
    save_path = './models/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_path)
    with open(save_path + 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    print('LOADING DATA...')

    # # IWSLT中文数据集**************************
    data_train = load_file('data/zh_iwslt/train')
    data_valid = load_file('data/zh_iwslt/valid')
    data_test = load_file('data/zh_iwslt/test_valid')
    # vocab.txt所在的位置

    # NOTE chinese-roberta-tokenizer
    pretrained = 'models/hfl/chinese-macbert-base'
    tokenizer = BertTokenizer.from_pretrained(
        pretrained, do_lower_case = True
    )

    print('PREPROCESSING DATA...')
    X_train, y_train = preprocess_data(
        data_train,
        tokenizer,
        punctuation_enc,
        seq_len,
    )
    X_valid, y_valid = preprocess_data(
        data_valid,
        tokenizer,
        punctuation_enc,
        seq_len,
    )

    print('INITIALIZING MODEL...')
    output_size = len(punctuation_enc)

    #  NOTE RobertaChineseLinearPunc
    bert_punc = BertPunc(output_size).cuda()

    print(bert_punc)

    best_val_loss = 1.e9
    print('TRAINING ALL LAYERS...')
    data_loader_train = create_data_loader(
        X_train,
        y_train,
        True,
        batch_size_all,
    )
    data_loader_valid = create_data_loader(
        X_valid,
        y_valid,
        False,
        batch_size_all,
    )
    for p in bert_punc.bert.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(bert_punc.parameters(), lr = learning_rate_all)

    print(optimizer)
    criterion = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(
        bert_punc,
        optimizer,
        criterion,
        epochs_all,
        data_loader_train,
        data_loader_valid,
        save_path,
        punctuation_enc,
        iterations_all,
        best_val_loss = best_val_loss
    )
