import time
import torch
import numpy as np
from torch.autograd import Variable
import lib.utils as utils
import torch.nn.functional as F



def train(loader, model, optimizer, logger, epoch, print_freq=10):
    # switch to train mode
    model.train()

    meters = logger.reset_meters('train')
    end = time.time()
    for i, sample in enumerate(loader):

        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        input_visual = Variable(sample['visual'].cuda())
        target = Variable(sample['label'].cuda(async=True))
        # compute output
        output =  model(input_visual)
        loss = F.cross_entropy(output, target)
        acc1, acc3 = utils.accuracy(output.data, target.data, topk=(1, 3))
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc3'].update(acc3[0], n=batch_size)
        meters['loss'].update(loss.data[0], n=batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Train]\tEpoch: [{0}][{1}/{2}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@3 {acc3.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc3=meters['acc3'], 
                   loss=meters['loss']))

    print('[Train]\tEpoch: [{0}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@3 {acc3.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc3=meters['acc3'], 
                   loss=meters['loss']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, logger, epoch=0, print_freq=100):
    # switch to train mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    for i, sample in enumerate(loader):

        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        input_visual = Variable(sample['visual'].cuda(), volatile=True)
        target = Variable(sample['label'].cuda(async=True), volatile=True)
        # compute output
        output =  model(input_visual)
        loss = F.cross_entropy(output, target)
        acc1, acc3 = utils.accuracy(output.data, target.data, topk=(1, 3))
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc3'].update(acc3[0], n=batch_size)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Val]\tEpoch: [{0}][{1}/{2}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@3 {acc3.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc3=meters['acc3'], 
                   loss=meters['loss']))

    print('[Val]\tEpoch: [{0}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@3 {acc3.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc3=meters['acc3'], 
                   loss=meters['loss']))

    logger.log_meters('val', n=epoch)

    return meters['acc1'].avg, meters['acc3'].avg


def evaluate(loader, model, logger, epoch=0, print_freq=100):

    # switch to train mode
    model.eval()
    meters = logger.reset_meters('test')
    end = time.time()
    results = []
    for i, sample in enumerate(loader):

        batch_size = sample['visual'].size(0)
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        input_visual = Variable(sample['visual'].cuda(), volatile=True)
        # compute output
        output =  model(input_visual)
        top3_result = output.cpu().data.topk(3, 1, largest=True, sorted=True)[0]
        for j in range(batch_size):
            result = {'image_id': sample['image_id'], 
                      'label_id': [top3_result[j].tolist()]}
            results.append(result)
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Evaluation] {0}/{1} batches processed'.format(i + 1, len(loader)))

    logger.log_meters('test', n=epoch)

    return results

    
