import argparse
import os
import shutil
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib import utils,logger, engine
import lib.datasets as datasets

# task specific package
import models
import pdb


parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/default.yaml', type=str, 
                    help='path to a yaml options file')
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument('--dir_logs', type=str, help='dir logs')
# data options
parser.add_argument('--trainsplit', type=str, choices=['train','trainval'])
# model options
parser.add_argument('--arch', type=str, default='resnet152', help='Model architecture. ')
# optim options
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')
parser.add_argument('--eval_epochs', type=int, default=10,
                    help='Number of epochs to evaluate the model')
# options not in yaml file          
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--save_model', default=True, type=utils.str2bool,
                    help='able or disable save model and optim state')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    help='print frequency')
################################################
parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                    help='show selected options before running')
# parser.add_argument('--finetuning_conv_epoch', type=int, default=10, help='From which epoch to finetuning the conv layers')

best_acc1 = 0.
best_acc3 = 0.



def main():
    global args, best_acc1, best_acc3
    args = parser.parse_args()

    # Set options
    options = {
        'data' : {
            'trainsplit': args.trainsplit, 
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'eval_epochs': args.eval_epochs,
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)

    
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)

    if args.help_opt:
        return

    # Set datasets
    print('Loading dataset....',)
    trainset = datasets.Scene(options['data']['trainsplit'], options['data'], is_testing=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=options['optim']['batch_size'],
                                        num_workers=args.workers,
                                        shuffle=True, pin_memory=True, drop_last=True)            
    valset = datasets.Scene('val', options['data'], is_testing=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=options['optim']['batch_size'],
                                    num_workers=args.workers, pin_memory=True)

    testset = datasets.Scene('test', options['data'], is_testing=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=options['optim']['batch_size'],
                                    num_workers=args.workers, pin_memory=True)
    print('Done.')
    print('Setting up the model...')

    # Set model, criterion and optimizer
    # assert options['model']['arch_resnet'] == options['coco']['arch'], 'Two [arch] should be set the same.'
    model = getattr(models, options['model']['arch'])(options['model'])

    # Optionally resume from a checkpoint
    exp_logger = None
    if args.resume:
        print('Loading saved model...')
        args.start_epoch, best_acc3, exp_logger = load_checkpoint(model, optimizer,#model.module, optimizer,
            os.path.join(options['logs']['dir_logs'], args.resume))
        model.features.set_trainable(True)
    else:
        # Or create logs directory
        if os.path.isdir(options['logs']['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'
                .format(options['logs']['dir_logs'], default=False)):
                os.system('rm -r ' + options['logs']['dir_logs'])
            else:
                return
        os.system('mkdir -p ' + options['logs']['dir_logs'])
        path_new_opt = os.path.join(options['logs']['dir_logs'],
                       os.path.basename(args.path_opt))
        path_args = os.path.join(options['logs']['dir_logs'], 'args.yaml')
        with open(path_new_opt, 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        print('Fix the pretrained parameters for one epoch')
        model.features.set_trainable(False)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=options['optim']['lr'], weight_decay=options['optim']['weight_decay'])

    if exp_logger is None:
        # Set loggers
        exp_name = os.path.basename(options['logs']['dir_logs']) # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        if options['data']['trainsplit'] == 'train':
            exp_logger.add_meters('val', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(exp_logger.info['model_params']))

    # Begin evaluation and training
    model = nn.DataParallel(model).cuda()
    if args.evaluate:
        print('Start evaluating...')
        path_logger_json = os.path.join(options['logs']['dir_logs'], 'logger.json')

        evaluate_result = engine.evaluate(test_loader, model, exp_logger, args.print_freq)
    
        save_results(evaluate_result, args.start_epoch, valset.split_name(),
                         options['logs']['dir_logs'])

        return

    print('Start training')
    for epoch in range(args.start_epoch, options['optim']['epochs']):
        #adjust_learning_rate(optimizer, epoch)selected_a.reinforce(reward.data.view(selected_a.size()))
        # train for one epoch
        # at first, the conv layers are fixed
        # if epoch > args.finetuning_conv_epoch and to_set_trainable:
        #     set_trainable(model.module.shared_conv_layer, True)
        #     optimizer = select_optimizer(
        #                     options['optim']['optimizer'], params=filter(lambda p: p.requires_grad, model.parameters()), 
        #                     lr=options['optim']['lr'], weight_decay=options['optim']['weight_decay'])
        #     to_set_trainable = False
        # optimizer = adjust_optimizer(optimizer, epoch, regime)
        engine.train(train_loader, model, optimizer,
                      exp_logger, epoch, args.print_freq) 

        if options['data']['trainsplit'] == 'train':
            # evaluate on validation set
            acc1, acc3 = engine.validate(val_loader, model,
                                                exp_logger, epoch, args.print_freq)
            if (epoch + 1) % options['optim']['eval_epochs'] == 0:
                #print('[epoch {}] evaluation:'.format(epoch))
                evaluate_result = engine.evaluate(test_loader, model, exp_logger, args.print_freq)   #model.module, exp_logger, args.print_freq)
                save_results(evaluate_result, epoch, valset.split_name(),
                         options['logs']['dir_logs'], is_testing=False)

            # remember best prec@1 and save checkpoint
            is_best = acc3 > best_acc3
            best_acc1 = acc1 if is_best else best_acc1
            best_acc3 = acc3 if is_best else best_acc3

            print('** [Best]\tAcc@1: {0:.2f}%\tAcc@3: {1:.2f}%'.format(
                best_acc1, best_acc3))
            save_checkpoint({
                    'epoch': epoch,
                    'arch': options['model']['arch'],
                    'best_acc1': best_acc1,
                    'best_acc3': best_acc3,
                    'exp_logger': exp_logger
                },
                model.module.state_dict(), #model.module.state_dict(),
                optimizer.state_dict(),
                options['logs']['dir_logs'],
                args.save_model,
                args.save_all_from,
                is_best)
        else:
            raise NotImplementedError

        if epoch == 0 and not args.resume:
            model.module.features.set_trainable(True)
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.module.parameters()), 
                    lr=options['optim']['lr'], weight_decay=options['optim']['weight_decay'])
    

def make_meters():  
    meters_dict = {
        'loss': logger.AvgMeter(),
        'batch_time': logger.AvgMeter(),
        'data_time': logger.AvgMeter(),
        'epoch_time': logger.SumMeter(), 
        'acc1': logger.AvgMeter(),
        'acc3': logger.AvgMeter(),
    }
    return meters_dict

def save_results(results, epoch, split_name, dir_logs, is_testing=True):
    if is_testing:
        subfolder_name = 'evaluate'
    else:
        subfolder_name = 'epoch_' + str(epoch)
    dir_epoch = os.path.join(dir_logs, subfolder_name)
    name_json = 'Scene_results_{}.json'.format(split_name)
    # TODO: simplify formating
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)

def save_checkpoint(info, model, optim, dir_logs, save_model, save_all_from=None, is_best=True):
    os.system('mkdir -p ' + dir_logs)
    if save_all_from is None:
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth.tar')
        path_best_info  = os.path.join(dir_logs, 'best_info.pth.tar')
        path_best_model = os.path.join(dir_logs, 'best_model.pth.tar')
        path_best_optim = os.path.join(dir_logs, 'best_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
    else:
        is_best = False # because we don't know the test accuracy
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_epoch,{}_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_epoch,{}_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_epoch,{}_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info['epoch']))
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info['epoch']))
            torch.save(optim, path_ckpt_optim.format(info['epoch']))
        if  info['epoch'] > 1 and info['epoch'] < save_all_from + 1:
            os.system('rm ' + path_ckpt_info.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_model.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_optim.format(info['epoch'] - 1))
    if not save_model:
        print('Warning train.py: checkpoint not saved')

def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.pth.tar'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc3   = 0
        exp_logger  = None
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_acc3' in info:
            best_acc3 = info['best_acc3']
        else:
            print('Warning train.py: no best_acc3 to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    #  if os.path.isfile(path_ckpt_optim):
    #      optim_state = torch.load(path_ckpt_optim)
    #      optimizer.load_state_dict(optim_state)
    #  else:
    #      print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_acc3 {})"
              .format(path_ckpt, start_epoch, best_acc3))
    return start_epoch, best_acc3, exp_logger

if __name__ == '__main__':
    main()
