import os
import itertools
import glob
import shutil
import time
import numpy as np
import torch
from tools.util import AttrDict, worker_init_fn
from torch.utils.data import DataLoader
from tools.vis import dataset_vis
from masks.multiblock import MaskCollator as MBMaskCollator
from masks.utils import apply_masks, repeat_interleave_batch

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

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
    name = "ijepa"  # exp_name
    checkpoints_dir = './checkpoints'
    epoch_count = 1
    batchsize = 4
    
    data_name = 'ABDOMINAL'
    tr_domain = 'MR'
    te_domain = 'CT'

    # data_name = 'MMS'
    # tr_domain = ['vendorA']
    # te_domain = ['vendorC']
    


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
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = MMS.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) 
        else:
            test_set        = MMS.get_test_all(modality = opt.te_domain, norm_func = None)
        label_name          = MMS.LABEL_NAME
    
    else:
        raise NotImplementedError 
    

    _log.info(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    # dataset_vis(test_set, save_path='CT', vis_num=10)     # dataset 可视化验证：数据和标签


    # Mask 构建
    mask_collator = MBMaskCollator(input_size=(192, 192))
    train_loader = DataLoader(dataset = train_set, num_workers = 4,\
                              batch_size = opt.batchsize, shuffle = True, 
                              drop_last = True, worker_init_fn =  worker_init_fn, 
                              pin_memory = True, collate_fn=mask_collator)
    
    for epoch in range(opt.epoch_count):
        epoch_start_time = time.time()
        for i, (train_batch, masks_enc, masks_pred) in enumerate(train_loader):
            train_input = {'img': train_batch["img"],
                            'lb': train_batch["lb"]}
            

            
            h = torch.rand((4, 144, 2048))
            h1 = apply_masks(h, masks_pred)
            h2 = repeat_interleave_batch(h1, opt.batchsize, repeat=len(masks_enc))

            
            
        











    