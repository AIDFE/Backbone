import os
import itertools
import glob
import shutil
import time
import numpy as np
import torch
from masks.multiblock import MaskCollator as MBMaskCollator
from tools.util import AttrDict, worker_init_fn, SoftDiceLoss
from torch.utils.data import DataLoader
from tools.vis import dataset_vis
from tools.test_dice import prediction_wrapper
from networks.smpmodels import efficient_unet
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tensorboardX import SummaryWriter

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('MedAI') 
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './my_utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config          # Sacred 相关
def cfg():

    seed = 1234
    name = "effectnet-unet-seg"  # exp_name
    checkpoints_dir = './checkpoints'
    snapshot_dir = ''
    epoch_count = 2000
    batchsize = 20
    infer_epoch_freq = 50
    save_epoch_freq = 50
    lr = 0.0003
    
    data_name = 'ABDOMINAL'
    tr_domain = 'MR'
    te_domain = 'CT'

    # data_name = 'MMS'
    # tr_domain = ['vendorA']
    # te_domain = ['vendorC']

    num_classes = 5
    patch_size = 16
    in_channels = 3


@ex.config_hook     # Sacred 相关
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{config["name"]}'
    observer = FileStorageObserver.create(os.path.join(config['checkpoints_dir'], exp_name))
    ex.observers.append(observer)
    return config


@ex.automain
def main(_run, _config, _log):
    # configs for sacred
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'), exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        _config['run_dir'] = _run.observers[0].dir
        _config['snapshot_dir'] = f'{_run.observers[0].dir}/snapshots'

    opt = AttrDict(_config)

     # load dataset
    if opt.data_name == 'ABDOMINAL':
        import dataloaders.abd_dataset as ABD
        if not isinstance(opt.tr_domain, list):
            opt.tr_domain = [opt.tr_domain]
            opt.te_domain = [opt.te_domain]

        train_set       = ABD.get_training(modality = opt.tr_domain)
        val_set         = ABD.get_validation(modality = opt.tr_domain, norm_func = train_set.normalize_op)
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = ABD.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) 
        else:
            test_set        = ABD.get_test_all(modality = opt.te_domain, norm_func = None)
        label_name          = ABD.LABEL_NAME
    
    elif opt.data_name == 'MMS':
        import dataloaders.mms_dataset as MMS
        if not isinstance(opt.tr_domain, list):
            opt.tr_domain = [opt.tr_domain]
            opt.te_domain = [opt.te_domain]

        train_set       = MMS.get_training(modality = opt.tr_domain)
        val_set         = MMS.get_validation(modality = opt.tr_domain, norm_func = train_set.normalize_op)
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = MMS.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) 
        else:
            test_set        = MMS.get_test_all(modality = opt.te_domain, norm_func = None)
        label_name          = MMS.LABEL_NAME
    
    else:
        raise NotImplementedError 
    

    _log.info(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    # dataset_vis(test_set, save_path='CT', vis_num=10)     # dataset 可视化验证：数据和标签

    train_loader = DataLoader(dataset = train_set, num_workers = 4,\
                              batch_size = opt.batchsize, shuffle = True, 
                              drop_last = True, worker_init_fn =  worker_init_fn, 
                              pin_memory = True)
    
    val_loader = DataLoader(dataset = val_set, num_workers = 4,\
                             batch_size = 1, shuffle = False, pin_memory = True)
    
    
    test_loader = DataLoader(dataset = test_set, num_workers = 4,\
                             batch_size = 1, shuffle = False, pin_memory = True)
    



    _log.info(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    # dataset_vis(test_set, save_path='CT', vis_num=10)     # dataset 可视化验证：数据和标签


    # Mask 构建
    mask_collator = MBMaskCollator(input_size=(192, 192))
    train_loader = DataLoader(dataset = train_set, num_workers = 4,\
                              batch_size = opt.batchsize, shuffle = True, 
                              drop_last = True, worker_init_fn =  worker_init_fn, 
                              pin_memory = True, collate_fn=mask_collator)
    

    model = efficient_unet(nclass = opt.num_classes, in_channel = 3)
    model.train()
    iter_num = 0
    max_iterations = opt.epoch_count * len(train_loader)
    best_score = 0
   

    
    for epoch in range(opt.epoch_count):
        epoch_start_time = time.time()
        for i, (train_batch, masks_enc, masks_pred) in enumerate(train_loader):
            img = train_batch['img'].cuda()
            lb = train_batch['lb'].cuda()

            # 可视化Mask
            




            # outputs, enc_feature = model(img)
            # enc_feature = enc_feature[-2]
            # h = enc_feature.reshape(opt.batchsize, enc_feature.shape[1], -1).permute((0, 2, 1))  # [B, HW, D]
            
            
           
            

            
            # h = torch.rand((4, 144, 2048))
            # h1 = apply_masks(h, masks_pred)
            # h2 = repeat_interleave_batch(h1, opt.batchsize, repeat=len(masks_enc))

            
            
        


 








    