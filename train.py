from utils.util import create_logger, AverageMeter, save_checkpoint, acc
import time
from criterion import CriterionAll
from utils.util import decode_parsing, inv_preprocess
from config import config
from networks.DinkNet_ASPP import DinkNet50
from networks.CPN_parsing import CDinkNet_ASPP
print('CPN')
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


from dataset.LIP_dataset_RGB import LIPDataSet
print('RGB dataset')
print(config.NUM_WORKERS)
# from networks.CDinkNet_ASPP import CDinkNet_ASPP


# -------------------------pre---------------------------------------
logger, out_dir, vis_dir = create_logger(config)
logger.info('initalize logger succesiful.')
writer = SummaryWriter(vis_dir)

cudnn.enabled = True
# cudnn related setting
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
gpus = [int(i) for i in config.GPUS.split(',')]
# ------------------------got data and model------------------------------------------
w, h = map(int, config.DATA.INPUT_SIZE.split(','))
input_size = [w, h]
heatmap_size = [w//4, h//4]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_loader = data.DataLoader(LIPDataSet(config.DIR.DATA, config.DATA.SPLIT, crop_size=input_size, heatmap_size=heatmap_size, transform=transform),
                               batch_size=config.DATA.BATCH_SIZE * len(gpus), shuffle=True, num_workers=config.NUM_WORKERS,
                               pin_memory=True)
logger.info('initalize and data succesiful.')

model = eval(config.MODEL.NAME)(config)
model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
logger.info('parallel and initialize model succesiful')
# ----------------------------got criterion and optimizer-----------------------------
# poly learning_rate


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_ter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(config.TRAIN.LR*len(gpus), i_ter,
                 total_iters, config.TRAIN.POLY_POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


criterion = CriterionAll().cuda()
optimizer = optim.SGD(
    model.parameters(),
    lr=config.TRAIN.LR*len(gpus),
    momentum=config.TRAIN.MOMENTUM,
    weight_decay=config.TRAIN.WEIGHT_DECAY
)
logger.info('initalize criterion and optimzer succesiful.')

if config.TRAIN.RESUME == True:
    model_state_file = config.TRAIN.PRE
    checkpoint = torch.load(model_state_file)
    if checkpoint['model'] == config.MODEL.NAME:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        begin_epoch = checkpoint['epoch']
        logger.info('upload model successiful,epoch:{},model:{},'.format(
            checkpoint['epoch'], checkpoint['model']))
else:
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    print(config.MODEL.NAME)

# ----------------------------train -----------------------------------------------------------
logger.info('model:{},epoch:{},gpus:{},learning_rate:{},batchsize:{}'.format(config.MODEL.NAME,
                                                                             config.TRAIN.END_EPOCH-begin_epoch, gpus, config.TRAIN.LR*len(gpus), config.DATA.BATCH_SIZE*len(gpus)))
epoch_end = time.time()
batch_time = AverageMeter()
losses = AverageMeter()
epoch_time = AverageMeter()
data_time = AverageMeter()
total_iters = config.TRAIN.END_EPOCH * len(train_loader)


for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
    model.train()
    end = time.time()
    parsing_acc = AverageMeter()
    for i_ter, (images, labels,heatmap,heatmap_weight, _) in enumerate(train_loader):
        data_time.update(time.time()-end)
        i_ter += len(train_loader) * epoch
        lr = adjust_learning_rate(optimizer, i_ter, total_iters)

#        fowrad
        input = images.cuda()
        labels = labels.long().cuda()

        preds = model(input)
        # backward
        optimizer.zero_grad()
        loss = criterion(preds, labels)
        preds = preds[0]
        loss.backward()
        optimizer.step()
        preds_acc = preds.detach().cpu()
        preds_acc = torch.nn.functional.interpolate(input=preds_acc, size=(h, w),
                                                    mode='bilinear', align_corners=True)


        losses.update(loss.item())
        batch_time.update(time.time()-end)
        end = time.time()
        # parsing_acc.update(batch_acc)

        if i_ter % 500 == 0:
            writer.add_scalar('learning_rate', lr, i_ter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_ter)
            batch_acc = acc(preds_acc.numpy(),
                        labels.detach().cpu().numpy())

            # images_inv = inv_preprocess(images, 2)
            # labels_colors = decode_parsing(labels, 2, 20, is_pred=False)

            # if isinstance(preds, list):
            #     preds = preds[0]
            # preds_colors = decode_parsing(preds, 2, 20, is_pred=True)
            # img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
            # lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
            # pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)

            # writer.add_image('Images/', img, i_ter)
            # writer.add_image('Labels/', lab, i_ter)
            # writer.add_image('Preds/', pred, i_ter)

            logger.info('epoch:{},{}/{}      learning_rate:{:}'.format(epoch,
                                                                       i_ter-epoch*len(train_loader), len(train_loader), lr))
            logger.info('loss:{:.2f}     accuracy:{}'.format(
                losses.val, batch_acc))
            logger.info('batch_time{:.2f},totle time{:.2f}'.format(
                batch_time.val, batch_time.sum))
            logger.info('data_time:{:.2f}'.format(data_time.avg))
            print('out_shape:{}'.format(preds.shape))
            parsing_acc = AverageMeter()
# information for epoch
    epoch_time.update(time.time()-epoch_end)
    epoch_end = time.time()
    logger.info('*******')
    logger.info('')
    logger.info('epoch:{}   total_time:{:.2f}.'.format(epoch, epoch_time.sum))
    logger.info('train_loss:{:.2f}'.format(losses.avg))
# save model for every epoch
    save_checkpoint({
        'epoch': epoch + 1,
        'model': config.MODEL.NAME,
        'state_dict': model.state_dict(),  # for resume
        'module_state_dict': model.module.state_dict(),  # for eval
        'optimizer': optimizer.state_dict(),
    }, out_dir, filename='checkpoint_parallel_deconv.pth')
    logger.info('save checkpoint in {}'.format(out_dir))
logger.info('finish')
