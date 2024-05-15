import os
import torch
from modules import SimCLR, LARS, ReCon32, ReCon64, ReCon224, HiCon, HiCon_r18, SlotCLR


def load_model(args, reload_model=False, load_path=None, data='non_imagenet'):

    if args.model == 'orig':
        model = SimCLR(args, data=data)
        # print(model)
        if reload_model:
            if os.path.isfile(load_path):
                model_fp = os.path.join(load_path)
            else:
                print("No file to load")
                return
            model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
        model = model.cuda()
        # print(model)
    
    if args.model == 'hicon':
        model = HiCon_r18(args, data=data)
        if reload_model:
            if os.path.isfile(load_path):
                model_fp = os.path.join(load_path)
            else:
                print("No file to load")
                return
            model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
        model = model.cuda()

    if args.model == 'hicon_dual':
        model = HiCon(args, data=data)
        if reload_model:
            if os.path.isfile(load_path):
                model_fp = os.path.join(load_path)
            else:
                print("No file to load")
                return
            model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
        model = model.cuda()
    
    recon = None
    params = model.parameters()

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == "LARS":
        # LearningRate=(0.3×BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(params, lr=learning_rate, weight_decay=args.weight_decay, exclude_from_weight_decay=["batch_normalization", "bias"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    else:
        raise NotImplementedError

    return model, recon, optimizer, scheduler


def save_model(model_dir, model, epoch):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))