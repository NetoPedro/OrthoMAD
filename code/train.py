# Imports
import os
import copy
import pandas as pd
import numpy as np
import csv
import logging
from tqdm import tqdm
from pathlib import Path

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Project Imports
from data_utils import FaceDataset
from metrics_utils import performances_compute, performances_compute2
from resnet_utils import Resnet34, Resnet18



# Function: Train Function
def train_fn(model, data_loader, data_size, optimizer, criterion, weight_loss, device):
    model.train()

    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_corrects = 0
    for i, (inputs, labels) in enumerate(tqdm(data_loader)):
        inputs, labels = inputs.to(device), torch.FloatTensor(labels *1.0).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)


        loss_1 = criterion(outputs[2], labels)
        loss_2 = weight_loss * torch.bmm(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], -1, 1)).reshape(outputs[2].shape[0]).pow(2).mean()
        loss =  loss_1+loss_2 

        _, preds = torch.max(outputs[2].reshape((-1,1)),dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_loss_1 += loss_1.item() * inputs.size(0)
        running_loss_2 += loss_2.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size
    epoch_loss_1 = running_loss_1 / data_size
    epoch_loss_2 = running_loss_2 / data_size
    epoch_acc = running_corrects.double() / data_size

    print('{} Loss: {:.4f} Loss C: {:.4f} Loss O: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss,epoch_loss_1,epoch_loss_2, epoch_acc))

    return epoch_loss, epoch_acc



# Function: Evaluation Function
def eval_fn(model, data_loader, data_size, criterion, device):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        prediction_scores, gt_labels = [], []
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), torch.FloatTensor(labels *1.0).to(device)

            outputs = model(inputs)
            loss= criterion(outputs[2], labels)

            _, preds = torch.max(outputs[2].reshape((-1,1)),dim=1)

            probs = outputs[2].reshape((-1,1))
            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # import pdb
        # pdb.set_trace()
        
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        auc, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
        _, bpcer01, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.001, verbose=False)
        _, bpcer1, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.01, verbose=False)
        _, bpcer10, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.1, verbose=False)
        _, bpcer20, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.2, verbose=False)


        print('{} Loss: {:.4f} Auc: {:.4f} EER: {:.4f} MinP {:.4f} MaxP {:.4f}'.format('Val', epoch_loss, auc, eer_value,min(prediction_scores),max(prediction_scores)))
        print('{} BPCER 0.1: {:.4f} BPCER 1.0: {:.4f} BPCER 10.0 {:.4f} BPCER 20.0 {:.4f}'.format('val',bpcer01,bpcer1,bpcer10,bpcer20))
    
    return epoch_loss, epoch_acc, eer_value, bpcer20, bpcer10, bpcer1, bpcer01



# Function: Run training
def run_training(model, model_path, device, logging_path, num_epochs, dataloaders, dataset_sizes, lr, weight_loss, output_name, earlystop_patience=50):
    model = model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer=optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    logging.basicConfig(filename=logging_path, level=logging.INFO)

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_eer = 100
    lowest_20 = 1000
    lowest_10 = 1000
    lowest_1 = 1000
    lowest_01 = 1000
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        # Each epoch has a training and validation phase
        train_loss, train_acc = train_fn(model, dataloaders['train'], dataset_sizes['train'], optimizer, criterion, weight_loss, device=device)
        val_loss, val_acc, val_eer_values,out_20,out_10,out_1,out_01 = eval_fn(model, dataloaders['val'], dataset_sizes['val'], criterion, device=device)
        logging.info('train loss: {}, train acc: {}, val loss: {}, val acc: {}, val eer: {}'.format(train_loss, train_acc, val_loss, val_acc, val_eer_values))

        # deep copy the model
        if val_eer_values <= lowest_eer:
            lowest_eer = val_eer_values
            lowest_20 = out_20
            lowest_10 = out_10
            lowest_1 = out_1
            lowest_01 = out_01
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == earlystop_patience or epoch >= num_epochs:
            early_stop = True
        else:
            continue

        if early_stop:
            print('Trian process Stopped')
            print('epoch: {}'.format(epoch))
            break

    print('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info(f'saved model path: {model_path}')

    with open("results_"+output_name, mode='a') as csv_file:
        fieldnames = ['lower_eer','01','1','10','20']
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerow([lowest_eer,lowest_01,lowest_1,lowest_10,lowest_20])

    # save best model weights
    torch.save(best_model_wts, model_path)



# Function: Run test
def run_test(test_loader, model, model_path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction_scores, gt_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = outputs[2]

            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

        std_value = np.std(prediction_scores)
        mean_value = np.mean(prediction_scores)
        prediction_scores = [ (float(i) - mean_value) /(std_value) for i in prediction_scores]
        _, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
        print(f'Test EER value: {eer_value*100}')

    return prediction_scores



# Function: Write scores to a .CSV file
def write_scores(test_csv, prediction_scores, output_path):
    save_data = []
    dataframe = pd.read_csv(test_csv)
    for idx in range(len(dataframe)):
        image_path = dataframe.iloc[idx, 0]
        label = dataframe.iloc[idx, 1]
        label = label.replace(' ', '')
        save_data.append({'image_path': image_path, 'label':label, 'prediction_score': prediction_scores[idx]})

    with open(output_path, mode='w') as csv_file:
        fieldnames = ['image_path', 'label', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in save_data:
            writer.writerow(data)

    print(f'Saving prediction scores in {output_path}.')



# Function: Main
def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    

    models = ["resnet18"]
    for model_name in models:
        if model_name == "resnet18":
            model = Resnet18(args.latent_size)
        elif model_name == "resnet34":
            model = Resnet34(args.latent_size)
        
        print(model)
        
        if args.is_train:
            train_dataset = FaceDataset(args.train_csv_path, is_train=True)
            test_dataset = FaceDataset(args.test_csv_path, is_train=False)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
            dataloaders = {'train': train_loader, 'val': test_loader}
            dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
            print('train and test length:', len(train_dataset), len(test_dataset))

            # compute loss weights to improve the unbalance between data
            attack_num, bonafide_num = 0, 0
            with open(args.train_csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['label'] == 'attack':
                        attack_num += 1
                    else:
                        bonafide_num += 1
            print('attack and bonafide num:', attack_num, bonafide_num)

            # nSamples  = [attack_num, bonafide_num]
            # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            # normedWeights = torch.FloatTensor(normedWeights).to(device)

            #create log file and train model
            logging_path = os.path.join(args.output_dir, 'train_info.log')
            run_training(model, args.model_path, logging_path, args.max_epoch, dataloaders, dataset_sizes,args.lr,args.weight_loss,output_name=model_name)

        if args.is_test:
            # create save directory and path
            test_output_folder = os.path.join(args.output_dir, 'test_results')
            Path(test_output_folder).mkdir(parents=True, exist_ok=True)
            test_output_path = os.path.join(test_output_folder, 'test_results.csv')
            # test
            test_dataset = FaceDataset(file_name=args.test_csv_path, is_train=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            test_prediction_scores = run_test(test_loader=test_loader, model=model, model_path=args.model_path, device=device)
            write_scores(args.test_csv_path, test_prediction_scores, test_output_path)



# Main
if __name__ == '__main__':

    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='MixFaceNet model')
    parser.add_argument("--train_csv_path", default="mor_gan_train.csv", type=str, help="input path of train csv")
    parser.add_argument("--test_csv_path", default="mor_gan_train.csv", type=str, help="input path of test csv")

    parser.add_argument("--output_dir", default="output", type=str, help="path where trained model and test results will be saved")
    parser.add_argument("--model_path", default="mixfacenet_SMDD", type=str, help="path where trained model will be saved or location of pretrained weight")

    parser.add_argument("--is_train", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="train database or not")
    parser.add_argument("--is_test", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="test database or not")

    parser.add_argument("--max_epoch", default=1, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
    parser.add_argument("--latent_size", default=64, type=int, help="train batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="train batch size")
    parser.add_argument("--weight_loss", default=1, type=float, help="train batch size")

    parser.add_argument("--gpu_id", type=int, default=3, help="The index of the GPU.")

    args = parser.parse_args()

    main(args)
