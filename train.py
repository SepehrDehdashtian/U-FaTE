import os
import sys
import time

import torch
import config
import traceback
from hal.utils import misc

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

import control
import hal.datasets as datasets

# import wandb

def main():
    # parse the arguments
    args = config.parse_args()

    if args.ngpu == 0:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    pl.seed_everything(args.manual_seed)

    # callbacks = [cbs.RichProgressBar()]
    callbacks = []
    if args.save_results:
        tb_logger = TensorBoardLogger(save_dir=args.logs_dir,
                                   log_graph=False,
                                   name=args.project_name + '_' + args.exp_name)
        wandb_logger = WandbLogger(save_dir=args.logs_dir,
                                #    log_graph=False,
                                   name=args.project_name + '_' + args.exp_name,
                                   group=args.wandb_group)
        
        logger = [tb_logger, wandb_logger]
        
        checkpoint = cbs.ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.project_name),
            filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
            monitor='val_loss',
            save_top_k=args.checkpoint_max_history,
            save_on_train_epoch_end=True,
            save_weights_only=True,
            save_last=None)
        enable_checkpointing = True
        callbacks.append(checkpoint)
    else:
        logger = False
        checkpoint = None
        enable_checkpointing = False

    torch.set_float32_matmul_precision('medium')

    dataloader = getattr(datasets, args.dataset)(args)
    model = getattr(control, args.control_type)(args, dataloader)

    if args.ngpu == 0:
        strategy = None
        sync_batchnorm = False
        accelerator = 'cpu'
    elif args.ngpu > 1:
        strategy = 'ddp'
        sync_batchnorm = True
        accelerator = 'gpu'
    else:
        strategy = 'dp'
        sync_batchnorm = False
        accelerator = 'gpu'
    
    torch.use_deterministic_algorithms(True)

    trainer = pl.Trainer(
                        #  gpus=args.ngpu,
                         devices=args.ngpu,
                         accelerator=accelerator,
                         strategy=strategy,
                         sync_batchnorm=sync_batchnorm,
                         enable_progress_bar=not args.no_progress_bar,
                         benchmark=True,
                         callbacks=callbacks,
                         enable_checkpointing=enable_checkpointing,
                         logger=logger,
                         log_every_n_steps=1,
                         min_epochs=1,
                         max_epochs=args.nepochs,
                         precision=args.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=args.check_val_every_n_epochs,
                        #  auto_scale_batch_size='power'
                         )
    # trainer.tune(model)
    trainer.fit(model)

    if args.test_flag.lower() == 'whole':
        trainer.test(model, dataloader.test_whole_dataset_dataloader())
    elif args.test_flag.lower() == 'test':    
        trainer.test(model, dataloader.test_dataloader())
    elif args.test_flag.lower() == 'val':
        trainer.test(model, dataloader.val_dataloader())


if __name__ == "__main__":
    misc.setup_graceful_exit()
    try:
        main()
    except KeyboardInterrupt:
        # do not print stack trace when ctrl-c is pressed
        traceback.print_exc(file=sys.stdout)
        misc.cleanup()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        misc.cleanup()
    
    time.sleep(3)