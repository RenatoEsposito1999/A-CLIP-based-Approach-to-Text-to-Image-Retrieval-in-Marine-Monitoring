# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate') #1e-4
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')#1e-4
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')#256
    parser.add_argument('--dataset_path', default="./dataset/train_dataset.csv", type=str, help='path to csv dataset') 
    parser.add_argument('--validation_path', default="./dataset/val_dataset.csv", type=str, help='path to csv dataset')
    parser.add_argument('--test_path', default="./dataset/test_dataset.csv", type=str, help='path to csv dataset')
    parser.add_argument('--only_turtle_validation_path', default="./dataset/only_turtle_val.csv", type=str, help='path to csv dataset with onlu turtle')            
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    parser.add_argument('--no_train', default=False, help='If true, training is not performed.')
    parser.add_argument('--test', default=False, help='If true, test is performed.')
    #parser.add_argument('--text_encoder', default="bert", help='The name of the text encoder to use as backbone')
    parser.add_argument('--best_model_mix_path', default="best_model_mix.pth", help='Path to best model on mix validation set')
    parser.add_argument('--best_model_turtle_only_path', default="best_model_only_turtle.pth", help='Path to best model on only turtle validation set')
    parser.add_argument('--resume_path', default="./checkpoint/", help='Checkpoint directory path')
    parser.add_argument('--metrics_path', default="./metrics/", help='Metrics directory path')
    parser.add_argument('--resume', default=False, help='If true, model start training from a checkpoint')
    parser.add_argument('--text_encoder', default="bert-base-uncased", help='Name of the text encoder')
    parser.add_argument('--vision_encoder', default="vit_base_patch16_224", help='Name of the ViT')
    parser.add_argument('--lora', default=False, help='If apply lora or not')
    args = parser.parse_args()
    return args