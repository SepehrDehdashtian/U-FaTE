import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

import hal.kernels as kernels
import hal.losses as losses
import hal.metrics as metrics
import hal.models as models
import hal.plugins as plugins
import hal.utils.misc as misc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ['BaseClass']

class BaseClass(pl.LightningModule):
    """
    This class will act as the base class for all the control methods.
    """
    def __init__(self, opts, dataloader):
        super().__init__()
        self.save_hyperparameters(opts)

        self.dataloader = dataloader

        self.val_dataloader = dataloader.val_dataloader
        self.train_dataloader = dataloader.train_dataloader
        if dataloader.test_dataloader:
            self.test_dataloader = dataloader.test_dataloader

        if self.hparams.dataset == 'BaseNodeEmbeddingBuiltIn' or hasattr(dataloader, 'dglG'):
            self.dglG = dataloader.dglG


        ## VLM model, if any
        if "vlm" in self.hparams.model_type.keys():
            self.vlm = getattr(models, self.hparams.model_type['vlm'])\
                (device=opts.device, **self.hparams.model_options['vlm'])
            
            # Insert the image transforms of the timm model to the dataloader
            self.dataloader.transform = self.vlm.transforms
            
        else:
            self.vlm = None


        ## TIMM model, if any
        if "timm" in self.hparams.model_type.keys():
            self.timm = getattr(models, self.hparams.model_type['timm'])\
                (device=opts.device, **self.hparams.model_options['timm'])
            
            # Insert the image transforms of the timm model to the dataloader
            self.dataloader.transform = self.timm.transforms
            
            if 'save_features' in self.hparams.model_options['timm'].keys():
                if self.hparams.model_options['timm']['save_features']:
                    self.hparams.timm_log_dir = self.hparams.dataset_options["path"]
        else:
            self.timm = None

        ## Feature Extractor model, if any
        if "feature_extractor" in self.hparams.model_type.keys():
            self.feature_extractor = getattr(models, self.hparams.model_type['feature_extractor'])(
                                    **self.hparams.model_options['feature_extractor'])
        else:
            self.feature_extractor = None

        ## Encoder model, if any
        if "encoder" in self.hparams.model_type.keys():
            self.encoder = getattr(models, self.hparams.model_type['encoder'])(
                                    **self.hparams.model_options['encoder'])
        else:
            self.encoder = None

        ## Target model, if any
        if "target" in self.hparams.model_type.keys():
            self.target = getattr(models, self.hparams.model_type['target'])(
                                    **self.hparams.model_options['target'])
        else:
            self.target = None

        ## Adversary model(s), if any
        if "adversary" in self.hparams.model_type.keys():
            self.adversary = getattr(models, self.hparams.model_type['adversary'])(
                                        **self.hparams.model_options['adversary'])

        elif "adversary_z" in self.hparams.model_type.keys():
            adversary_z = getattr(models, self.hparams.model_type['adversary_z'])(
                                    **self.hparams.model_options['adversary_z'])
            adversary_s = getattr(models, self.hparams.model_type['adversary_s'])(
                                    **self.hparams.model_options['adversary_s'])

            self.adversary = models.HGRAdversary(adversary_s, adversary_z)
        else:
            self.adversary = None

        ## Laplacian model(s), if any
        if "laplacian" in self.hparams.model_type.keys():
            self.laplacian = getattr(models, self.hparams.model_type['laplacian'])(L=dataloader.return_laplacian(), **self.hparams.model_options['laplacian'])
        else:
            self.laplacian = None

        ## Loss function
        self.criterion = nn.ModuleDict()
        

        if "target" in self.hparams.loss_type.keys():
            self.criterion['target'] = getattr(losses, self.hparams.loss_type['target'])(
                                            **self.hparams.loss_options['target'])
        else:
            self.criterion['target'] = losses.IdentityLoss()

        if "DEP_ZY" in self.hparams.loss_type.keys():
            self.criterion['DEP_ZY'] = getattr(losses, self.hparams.loss_type['DEP_ZY'])(
                                            **self.hparams.loss_options['DEP_ZY'])
        else:
            self.criterion['DEP_ZY'] = losses.IdentityLoss()


        if "DEP_ZS" in self.hparams.loss_type.keys():
            self.criterion['DEP_ZS'] = getattr(losses, self.hparams.loss_type['DEP_ZS'])(
                                            **self.hparams.loss_options['DEP_ZS'])
        else:
            self.criterion['DEP_ZS'] = losses.IdentityLoss()


        if "adversary" in self.hparams.loss_type.keys():
            self.criterion['adversary'] = getattr(losses, self.hparams.loss_type['adversary'])(
                                                    **self.hparams.loss_options['adversary'])
        else:
            self.criterion['adversary'] = losses.IdentityLoss()

        ## Metrics on target
        self.metric_target = nn.ModuleDict()
        self.metric_target["trn"] = nn.ModuleDict()
        self.metric_target["val"] = nn.ModuleDict()
        self.metric_target["test"] = nn.ModuleDict()

        if isinstance(self.hparams.metric_target_options, dict):
            for met_name, met_opts in self.hparams.metric_target_options.items():
                self.metric_target["trn"][met_name] = getattr(metrics, self.hparams.metric_target[met_name])(
                                                                **met_opts)
                self.metric_target["val"][met_name] = getattr(metrics, self.hparams.metric_target[met_name])(
                                                                **met_opts)
                self.metric_target["test"][met_name] = getattr(metrics, self.hparams.metric_target[met_name])(
                                                                **met_opts)

        ## Metrics on control
        self.metric_control = nn.ModuleDict()
        self.metric_control["trn"] = nn.ModuleDict()
        if isinstance(self.hparams.metric_control_options, dict):
            for met_name, met_opts in self.hparams.metric_control_options.items():
                self.metric_control["trn"][met_name] = getattr(metrics, self.hparams.metric_control[met_name])(**met_opts)

        self.metric_control["val"] = nn.ModuleDict()
        if isinstance(self.hparams.metric_control_options, dict):
            for met_name, met_opts in self.hparams.metric_control_options.items():
                self.metric_control["val"][met_name] = getattr(metrics, self.hparams.metric_control[met_name])(**met_opts)

        self.metric_control["test"] = nn.ModuleDict()
        if isinstance(self.hparams.metric_control_options, dict):
            for met_name, met_opts in self.hparams.metric_control_options.items():
                self.metric_control["test"][met_name] = getattr(metrics, self.hparams.metric_control[met_name])(**met_opts)

        self.automatic_optimization = False

    def preprocess_batch(self, batch):
        if isinstance(batch, dict):
            x_s, y_s, s_s = batch['source']
            x_t, y_t, s_t = batch['target']
            x = torch.cat((x_s, x_t), dim=0)
            y = torch.cat((y_s, y_t), dim=0)
            s = torch.cat((s_s, s_t), dim=0)
            mask = torch.cat((torch.zeros(s_s.size(0)), torch.ones(s_t.size(0))), dim=0)
        else:
            if len(batch) == 6:
                x_s, x_t, y_s, y_t, s_s, s_t = batch
                x = torch.cat((x_s, x_t), dim=0)
                y = torch.cat((y_s, y_t), dim=0)
                s = torch.cat((s_s, s_t), dim=0)
                mask = torch.cat((torch.zeros(s_s.size(0)), torch.ones(s_t.size(0))), dim=0)
            elif len(batch) == 4:
                x, y, s, mask = batch
                x_s = x[mask == 0]
                y_s = y[mask == 0]
                x_t = x[mask == 1]
                y_t = y[mask == 1]
            elif len(batch) == 3:
                x, y, s = batch
                mask = torch.zeros(x.size(0))
            else:
                raise ValueError("There is some problem in dataloader")

        return x, y, s, mask

    def turn_off_grad(self, model):
        for p in model.parameters():
            p.requires_grad = False

        return model

    def turn_on_grad(self, model):
        for p in model.parameters():
            p.requires_grad = True

        return model

    def conditional_process(self, val, y):
        # Return only those val where y == 1 if conditional_fairness is True

        # This is only for binary classification case
        if isinstance(val, (tuple, list)):
            return [self.conditional_process(v, y) for v in val]

        if self.hparams.conditional_fairness:
            return val[y[:, 1] == 1]
        else:
            return val

    def training_step_end(self, step_output):
        returned = list(step_output.keys())
        
        # import pdb; pdb.set_trace()
        
        loss = step_output["loss"]#['value']
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)

        if "tau" in returned:
            self.log('tau', step_output["tau"], on_step=True, on_epoch=False, prog_bar=False)


        for key in ['loss_tgt', 'loss_tgt_T', 'loss_tgt_S', 'loss_ctl', 'loss_laplacian', 'dep_zy', 'dep_zs', 'proj_y', 'proj_s']:
            if key in returned:
                if isinstance(step_output[key], dict):
                    loss = step_output[key]["value"]
                else:
                    loss = step_output[key]

                self.log(f'train_{key}', loss, on_step=False, on_epoch=True, prog_bar=True)


        if all([_ in returned for _ in ['y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]

            # This means we are not in DA setting
            for met_name, met_tgt in self.metric_target["trn"].items():
                if not "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y_hat, y)

        if all([_ in returned for _ in ['y_hat', 'y', 's']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]
            s = step_output["s"]
           
            for met_name, met_tgt in self.metric_target["trn"].items():
                if "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y, y_hat, s)

        if all([_ in returned for _ in ['s', 'z']]):
            z = step_output["z"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "alpha_beta" in met_name or "alpha_lin" in met_name or "lin_beta" in met_name or "lin_lin" in met_name or "DEP_ZS" in met_name:
                    met_ctl.update(z, s)

        if all([_ in returned for _ in ['x', 's']]):
            x = step_output["x"]
            s = step_output["s"]
            for met_name, met_ctl in self.metric_control["trn"].items():
                if "XS" in met_name or "DEP_XS" in met_name:
                    met_ctl.update(x,s)

        if all([_ in returned for _ in ['x', 'y']]):
            x = step_output["x"]
            y = step_output["y"]
            for met_name, met_ctl in self.metric_control["trn"].items():
                if "XY" in met_name or "DEP_XY" in met_name:
                    met_ctl.update(x,y)

        if all([_ in returned for _ in ['s', 'y']]):
            s = step_output["s"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "SY" in met_name:
                    met_ctl.update(s,y)

        if all([_ in returned for _ in ['z', 'y']]):
            z = step_output["z"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "DEP_ZY" in met_name or "ZY" in met_name:
                    met_ctl.update(z,y)

        if all([_ in returned for _ in ['s', 'y_hat']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "dpv" in met_name:
                    if "utility" in met_name:
                        met_ctl.update(y_hat, y, s)
                    else:
                        met_ctl.update(y_hat, s)
                
                if "SP" in met_name:
                    met_ctl.update(y_hat, s)

        if all([_ in returned for _ in ['s', 's_hat']]):
            s_hat = step_output["s_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "utility" in met_name:
                    met_ctl.update(s_hat, s)

        if all([_ in returned for _ in ['s', 'y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["trn"].items():
                if "EO" in met_name or "Medical" in met_name:
                    met_ctl.update(y_hat, y, s)
                if "SY" in met_name:
                    met_ctl.update(s,y)


        # Call scheduler, if we are on the last batch
        schs = self.lr_schedulers()
        if self.trainer.is_last_batch and schs is not None:
            if isinstance(schs, (list, tuple)):
                for i, s in enumerate(schs):
                    if self.used_optimizers[i]:
                        s.step()
                        self.used_optimizers[i] = False
            else:
                schs.step()


    def weighted_avg_loss(self, outputs):
        log_dict  = dict()
        tot_numel = dict()
        # Compute weighted average of epoch

        weighted_sum = dict()
        
        for o in outputs:
            for loss_key in ['loss_tgt', 'loss_ctl', 'loss_laplacian', 'dep_zy', 'dep_zs']:
                if loss_key in o:
                    if isinstance(o[loss_key], dict):
                            weighted_sum[loss_key]  = weighted_sum.get(loss_key, 0) + o[loss_key]['value'].item() * o[loss_key]['numel'].item()
                            tot_numel[loss_key] = tot_numel.get(loss_key, 0) + o[loss_key]['numel'].item()
                    else:
                        weighted_sum[loss_key]  = weighted_sum.get(loss_key, 0) + o[loss_key].item()
                        tot_numel[loss_key] = tot_numel.get(loss_key, 0) + 1

        for k, v in weighted_sum.items():
            log_dict[k] = v / tot_numel[k]

        return log_dict


    def training_epoch_end(self, outputs):

        log_dict = self.weighted_avg_loss(outputs)

        if self.hparams.log_z: self.log_z_s_y(outputs, 'train')

        # If the flags are True, save the features, labels, and decisions to 
        # a file in the dataset directory with the specific model name
        if "timm" in self.hparams.model_type.keys():
            if 'save_features' in self.hparams.model_options['timm'].keys():
                    if self.hparams.model_options['timm']['save_features']:
                        self.timm_log_data(outputs=outputs, split_name='train')
        

        log_dict["epoch"] = self.current_epoch

        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "train_loss_log.csv")

        misc.dump_log_dict(log_dict, log_file)

        if self.isLastValEpoch():
            self.add_to_results_dict(log_dict, 'train')


        log_dict = {}

        for met_name, met_tgt in self.metric_target["trn"].items():
            score = met_tgt.compute()
            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log("train_tgt_" + sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["tgt_"+sc_id + "_" + met_name] = sc_val.item() if torch.is_tensor(sc_val) else sc_val
            else:
                self.log("train_tgt_" + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["tgt_" + met_name] = score.item()
            met_tgt.reset()

        for met_name, met_ctl in self.metric_control["trn"].items():
            try:
                # print('train', met_name)
                score = met_ctl.compute()
            except Exception as e:
                print(e)
                print('Error in met_ctl.compute() :: base.py:training_epoch_end')
                import pdb; pdb.set_trace()

            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log('train_ctl_' + sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["ctl_" + sc_id + "_" + met_name] = sc_val.item()
            else:
                self.log('train_ctl_' + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["ctl_" + met_name] = score.item()
            met_ctl.reset()

        if self.isLastValEpoch():
            self.add_to_results_dict(log_dict, 'train')

            if self.hparams.test_flag is None: # If test_flag is not set, then we are in the final training/val and we need to dump the results into the csv file. Otherwise, we will dump the results in the test_epoch_end
                self.dump_results_dict()

        log_dict["epoch"] = self.current_epoch
        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "train_metric_log.csv")

        misc.dump_log_dict(log_dict, log_file)

        if self.hparams.no_progress_bar:
            print("Training epoch {} finished".format(self.current_epoch))

        # if self.hparams.log_z:


        if not self.hparams.build_kernel is None and self.hparams.control_type == 'EndToEndKIRL' and self.current_epoch <= self.hparams.pretrain_epochs:
            x_list = list()
            y_list = list()
            s_list = list()
            

            x_list = list(map(lambda o: o['x'], outputs))
            y_list = list(map(lambda o: o['y'], outputs))
            s_list = list(map(lambda o: o['s'], outputs))

            X = torch.cat(x_list, dim=0)
            Y = torch.cat(y_list, dim=0)
            S = torch.cat(s_list, dim=0)

            self.compute_kernel(features=X, Y=Y, S=S)


    def log_z_s_y(self, outputs, split_name):
        hasZ = bool(sum(list(map(lambda o: 'z' in o.keys(), outputs))))
        if hasZ:
            z_list = list(map(lambda o: o['z'], outputs))
        
        hasY = bool(sum(list(map(lambda o: 'y' in o.keys(), outputs))))
        if hasY:
            y_list = list(map(lambda o: o['y'], outputs))
        
        hasS = bool(sum(list(map(lambda o: 's' in o.keys(), outputs))))
        if hasS:
            s_list = list(map(lambda o: o['s'], outputs))

        hasY_hat = bool(sum(list(map(lambda o: 'y_hat' in o.keys(), outputs))))
        if hasY_hat:
            yhat_list = list(map(lambda o: o['y_hat'], outputs))

        hasS_hat = bool(sum(list(map(lambda o: 's_hat' in o.keys(), outputs))))
        if hasS_hat:
            shat_list = list(map(lambda o: o['s_hat'], outputs))

        hasHate = bool(sum(list(map(lambda o: 'hate' in o.keys(), outputs))))
        if hasHate:
            hate_list = list(map(lambda o: o['hate'], outputs))
        
        out_dir = os.path.join(self.hparams.logs_dir, 'outputs')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
  
        if hasZ:
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_z_{split_name}.out'), torch.cat(z_list, dim=0).cpu().numpy(), fmt='%10.5f')
        
        if hasY:
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_y_{split_name}.out'), torch.cat(y_list, dim=0).cpu().numpy(), fmt='%10.5f')
            
        if hasS:    
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_s_{split_name}.out'), torch.cat(s_list, dim=0).cpu().numpy(), fmt='%10.5f')
            
        if hasY_hat:
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_Yhat_{split_name}.out'), torch.cat(yhat_list, dim=0).cpu().numpy(), fmt='%10.5f')

        if hasS_hat:
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_Shat_{split_name}.out'), torch.cat(shat_list, dim=0).cpu().numpy(), fmt='%10.5f')

        if hasHate:
            np.savetxt(os.path.join(out_dir, f'{self.current_epoch}_Hate_{split_name}.out'), torch.cat(hate_list, dim=0).cpu().numpy(), fmt='%10.5f')


    def timm_log_data(self, outputs, split_name):
        data = dict()

        for key in ['z', 'y', 's', 'y_hat', 's_hat']:
            try:
                data[key] = torch.cat(list(map(lambda o: o[key], outputs)), dim=0).cpu().numpy()
            except KeyError:
                continue
        

        # Train a logistic regression model on the data or predict the labels using it.
        try:
            if self.hparams.model_options["timm"]["lin_logistic_regression"]:
                if split_name == 'train':
                    self.clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
                    # self.clf = LogisticRegression(solver='lbfgs')
                    self.clf.fit(data['z'], data['y'])
                # elif split_name in ['val', 'test']:
                data['y_hat'] = self.clf.predict(data['z'])
        except KeyError:
            pass

        
        # Save these data in the dataset directory in hal-datastore
        out_dir = os.path.join(self.hparams.timm_log_dir, 'timm_features', self.hparams.model_options["timm"]["model_name"])
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        np.savez_compressed(os.path.join(out_dir, f'{split_name}.npz'), **data)


    def validation_step_end(self, step_output):
        returned = list(step_output.keys())
        loss = step_output["loss"]
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)


        for key in ['loss_tgt', 'loss_tgt_T', 'loss_tgt_S', 'loss_ctl', 'loss_laplacian', 'dep_zy', 'dep_zs', 'proj_y', 'proj_s']:
            if key in returned:
                if isinstance(step_output[key], dict):
                    loss = step_output[key]["value"]
                else:
                    loss = step_output[key]

                self.log(f'val_{key}', loss, on_step=False, on_epoch=True, prog_bar=True)

        if all([_ in returned for _ in ['s', 's_hat']]):
            if isinstance(step_output["loss_ctl"], dict):
                loss_ctl = step_output["loss_ctl"]["value"]
            else:
                loss_ctl = step_output["loss_ctl"]
            self.log('val_loss_ctl', loss_ctl, on_step=False, on_epoch=True, prog_bar=True)


        if all([_ in returned for _ in ['s', 's_hat']]):
            adv_out = step_output["s_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["val"].items():
                if type(met_ctl).__name__ == "UtilityFromMSE":
                    met_ctl.update(adv_out, s)
                if "utility" in met_name:
                    met_ctl.update(adv_out, s)


        if all([_ in returned for _ in ['y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]

            # This means we are not in DA setting
            for met_name, met_tgt in self.metric_target["val"].items():
                if not "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y_hat, y)

        if all([_ in returned for _ in ['y_hat', 'y', 's']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]
            s = step_output["s"]
           
            for met_name, met_tgt in self.metric_target["val"].items():
                if "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y, y_hat, s)


        if all([_ in returned for _ in ['s', 'y_hat']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["val"].items():
                if "dpv" in met_name:
                    if "utility" in met_name:
                        met_ctl.update(y_hat, y, s)
                    else:
                        met_ctl.update(y_hat, s)
                
                if "SP" in met_name:
                    met_ctl.update(y_hat, s)


        if all([_ in returned for _ in ['s', 'y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["val"].items():
                if "EO" in met_name or "Medical" in met_name:
                    met_ctl.update(y_hat, y, s)
                if "SY" in met_name:
                    met_ctl.update(s,y)

        if all([_ in returned for _ in ['s', 'z']]):
            z = step_output["z"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["val"].items():
                
                if "alpha_beta" in met_name or "alpha_lin" in met_name or "lin_beta" in met_name or "lin_lin" in met_name or "DEP_ZS" in met_name:
                    met_ctl.update(z, s)


        if all([_ in returned for _ in ['z', 'y']]):
            z = step_output["z"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["val"].items():
                if "DEP_ZY" in met_name or "ZY" in met_name:
                    met_ctl.update(z,y)

        if all([_ in returned for _ in ['x', 's']]):
            x = step_output["x"]
            s = step_output["s"]
            for met_name, met_ctl in self.metric_control["val"].items():
                if "XS" in met_name or "DEP_XS" in met_name:
                    met_ctl.update(x,s)

        if all([_ in returned for _ in ['x', 'y']]):
            x = step_output["x"]
            y = step_output["y"]
            for met_name, met_ctl in self.metric_control["val"].items():
                if "XY" in met_name or "DEP_XY" in met_name:
                    met_ctl.update(x,y)


    def validation_epoch_end(self, outputs):
        log_dict = self.weighted_avg_loss(outputs)

        if self.hparams.log_z: self.log_z_s_y(outputs, 'val')

        # If the flags are True, save the features, labels, and decisions to 
        # a file in the dataset directory with the specific model name
        if "timm" in self.hparams.model_type.keys():
            if 'save_features' in self.hparams.model_options['timm'].keys():
                    if self.hparams.model_options['timm']['save_features']:
                        self.timm_log_data(outputs=outputs, split_name='val')


        log_dict["epoch"] = self.current_epoch
        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma
        if hasattr(self.hparams, 'amc_s') and self.hparams.amc_s is not None:
            log_dict["amc_s"] = self.hparams.amc_s
        if hasattr(self.hparams, 'amc_y') and self.hparams.amc_y is not None:
            log_dict["amc_y"] = self.hparams.amc_y
        if hasattr(self.hparams, 'kernel_x_options') and 'rff_dim' in self.hparams.kernel_x_options.keys():
            log_dict["drff"] = self.hparams.kernel_x_options['rff_dim']
        if hasattr(self.hparams, 'batch_size_train'):
            log_dict["batch_size"] = self.hparams.batch_size_train
        

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "val_loss_log.csv")

        misc.dump_log_dict(log_dict, log_file)

        if self.isLastValEpoch():
            self.add_to_results_dict(log_dict, 'val')

        log_dict = {}

        for met_name, met_tgt in self.metric_target["val"].items():
            # print('val', met_name)
            score = met_tgt.compute()
            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log("val_tgt_"+sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["tgt_"+sc_id + "_" + met_name] = sc_val.item() if torch.is_tensor(sc_val) else sc_val
            else:
                self.log("val_tgt_" + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["tgt_" + met_name] = score.item()
            met_tgt.reset()

        for met_name, met_ctl in self.metric_control["val"].items():
            # print(met_name)
            try:
                # print('val', met_name)
                score = met_ctl.compute()
            except Exception as e:
                print(e)
                print('Error in met_ctl.compute() :: base.py:validation_epoch_end')
                import pdb; pdb.set_trace()

            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log('val_ctl_'+sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["ctl_"+sc_id + "_" + met_name] = sc_val.item()
            else:
                self.log('val_ctl_' + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["ctl_"+met_name] = score.item()
            met_ctl.reset()
        

        if self.isLastValEpoch():
            self.add_to_results_dict(log_dict, 'val')



        log_dict["epoch"] = self.current_epoch
        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "val_metric_log.csv")

        misc.dump_log_dict(log_dict, log_file)



    def test_step_end(self, step_output):   

        returned = list(step_output.keys())
        loss = step_output["loss"]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)


        for key in ['loss_tgt', 'loss_tgt_T', 'loss_tgt_S', 'loss_ctl', 'loss_laplacian', 'dep_zy', 'dep_zs', 'proj_y', 'proj_s']:
            if key in returned:
                if isinstance(step_output[key], dict):
                    loss = step_output[key]["value"]
                else:
                    loss = step_output[key]

                self.log(f'test_{key}', loss, on_step=False, on_epoch=True, prog_bar=True)

        if all([_ in returned for _ in ['s', 's_hat']]):
            if isinstance(step_output["loss_ctl"], dict):
                loss_ctl = step_output["loss_ctl"]["value"]
            else:
                loss_ctl = step_output["loss_ctl"]
            self.log('test_loss_ctl', loss_ctl, on_step=False, on_epoch=True, prog_bar=True)


        if all([_ in returned for _ in ['s', 's_hat']]):
            adv_out = step_output["s_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["test"].items():
                if type(met_ctl).__name__ == "UtilityFromMSE":
                    met_ctl.update(adv_out, s)
                if "utility" in met_name:
                    met_ctl.update(adv_out, s)

        if all([_ in returned for _ in ['y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]

            # This means we are not in DA setting
            for met_name, met_tgt in self.metric_target["test"].items():
                if not "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y_hat, y)

        if all([_ in returned for _ in ['y_hat', 'y', 's']]):
            y_hat = step_output["y_hat"]
            y = step_output["y"]
            s = step_output["s"]
           
            for met_name, met_tgt in self.metric_target["test"].items():
                if "SGAccuracy" in met_name:
                    if len(y) == 0: print(met_tgt)
                    if len(y_hat) == 0: print(met_tgt)
                    met_tgt.update(y, y_hat, s)


        if all([_ in returned for _ in ['s', 'y_hat']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["test"].items():
                if "dpv" in met_name:
                    if "utility" in met_name:
                        met_ctl.update(y_hat, y, s)
                    else:
                        met_ctl.update(y_hat, s)
                
                if "SP" in met_name:
                    met_ctl.update(y_hat, s)


        if all([_ in returned for _ in ['s', 'y_hat', 'y']]):
            y_hat = step_output["y_hat"]
            s = step_output["s"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["test"].items():
                if "EO" in met_name or "Medical" in met_name:
                    met_ctl.update(y_hat, y, s)
                if "SY" in met_name:
                    met_ctl.update(s,y)

        if all([_ in returned for _ in ['s', 'z']]):
            z = step_output["z"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["test"].items():
                # if type(met_ctl).__name__ != "UtilityFromMSE" and "dpv" not in met_name and "EO" not in met_name and "SP" not in met_name and "utility" not in met_name and "XS" not in met_name and "SY" not in met_name:
                if "alpha_beta" in met_name or "alpha_lin" in met_name or "lin_beta" in met_name or "lin_lin" in met_name or "DEP_ZS" in met_name:
                    met_ctl.update(z, s)


        if all([_ in returned for _ in ['z', 'y']]):
            z = step_output["z"]
            y = step_output["y"]

            for met_name, met_ctl in self.metric_control["test"].items():
                if "DEP_ZY" in met_name or "ZY" in met_name:
                    met_ctl.update(z,y)

        if all([_ in returned for _ in ['hate', 's']]):
            hate = step_output["hate"]
            s = step_output["s"]

            for met_name, met_ctl in self.metric_control["test"].items():
                    met_ctl.update(hate, s)

        if all([_ in returned for _ in ['x', 's']]):
            x = step_output["x"]
            s = step_output["s"]
            for met_name, met_ctl in self.metric_control["test"].items():
                if "XS" in met_name or "DEP_XS" in met_name:
                    met_ctl.update(x,s)

        if all([_ in returned for _ in ['x', 'y']]):
            x = step_output["x"]
            y = step_output["y"]
            for met_name, met_ctl in self.metric_control["test"].items():
                if "XY" in met_name or "DEP_XY" in met_name:
                    met_ctl.update(x,y)


    def test_epoch_end(self, outputs):

        log_dict = self.weighted_avg_loss(outputs)

        if self.hparams.log_z: self.log_z_s_y(outputs, 'test')

        # If the flags are True, save the features, labels, and decisions to 
        # a file in the dataset directory with the specific model name
        if "timm" in self.hparams.model_type.keys():
            if 'save_features' in self.hparams.model_options['timm'].keys():
                    if self.hparams.model_options['timm']['save_features']:
                        self.timm_log_data(outputs=outputs, split_name='test')


        log_dict["epoch"] = self.current_epoch
        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma
        if hasattr(self.hparams, 'amc_s') and self.hparams.amc_s is not None:
            log_dict["amc_s"] = self.hparams.amc_s
        if hasattr(self.hparams, 'amc_y') and self.hparams.amc_y is not None:
            log_dict["amc_y"] = self.hparams.amc_y
        if hasattr(self.hparams, 'kernel_x_options') and 'rff_dim' in self.hparams.kernel_x_options.keys():
            log_dict["drff"] = self.hparams.kernel_x_options['rff_dim']
        if hasattr(self.hparams, 'batch_size_train'):
            log_dict["batch_size"] = self.hparams.batch_size_train
        

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "test_loss_log.csv")

        misc.dump_log_dict(log_dict, log_file)

        self.add_to_results_dict(log_dict, 'test')

        log_dict = {}

        for met_name, met_tgt in self.metric_target["test"].items():
            # print('val', met_name)
            score = met_tgt.compute()
            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log("test_tgt_"+sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["tgt_"+sc_id + "_" + met_name] = sc_val.item() if torch.is_tensor(sc_val) else sc_val
            else:
                self.log("test_tgt_" + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["tgt_" + met_name] = score.item()
            met_tgt.reset()

        for met_name, met_ctl in self.metric_control["test"].items():
            # print(met_name)
            try:
                # print('val', met_name)
                score = met_ctl.compute()
            except Exception as e:
                print(e)
                print('Error in met_ctl.compute() :: base.py:test_epoch_end')
                import pdb; pdb.set_trace()

            if isinstance(score, dict):
                for sc_id, sc_val in score.items():
                    self.log('test_ctl_'+sc_id + "_" + met_name, sc_val, on_step=False, on_epoch=True, prog_bar=True)
                    log_dict["ctl_"+sc_id + "_" + met_name] = sc_val.item()
            else:
                self.log('test_ctl_' + met_name, score, on_step=False, on_epoch=True, prog_bar=True)
                log_dict["ctl_"+met_name] = score.item()
            met_ctl.reset()
        

        self.add_to_results_dict(log_dict, 'test')
        self.dump_results_dict(test=True)


        log_dict["epoch"] = self.current_epoch
        if "tau" in outputs[-1].keys():
            log_dict["tau"] = outputs[-1]["tau"].item()
        elif self.hparams.tau is not None:
            log_dict["tau"] = self.hparams.tau
        if self.hparams.eps is not None:
            log_dict["eps"] = self.hparams.eps
        if self.hparams.beta is not None:
            log_dict["beta"] = self.hparams.beta
        if self.hparams.alpha is not None:
            log_dict["alpha"] = self.hparams.alpha
        if self.hparams.gamma is not None:
            log_dict["gamma"] = self.hparams.gamma

        log_dict["seed"] = self.hparams.manual_seed
        log_file = os.path.join(self.hparams.logs_dir, "test_metric_log.csv")

        misc.dump_log_dict(log_dict, log_file)



    def add_to_results_dict(self, inp_dict, split):
        if not hasattr(self, 'results_dict'):
            self.results_dict = dict()

        if not split in self.results_dict.keys():
            self.results_dict[split] = dict()

        for k, v in inp_dict.items():
            self.results_dict[split][k] = v

    def dump_results_dict(self, test=False):

        if not test and not "timm" in self.hparams.model_type.keys():
            # Do not log in the results.csv when the experiment is a tuning process
            if not self.hparams.tune_flag:
                results_file = os.path.join(self.hparams.result_path, 'results.csv')
                edited_dict = dict()
                edited_dict['seed'] = self.results_dict['train']['seed']
                edited_dict['tau'] = self.results_dict['train']['tau']
                edited_dict['beta'] = self.results_dict['train']['beta']
                edited_dict['alpha'] = self.results_dict['train']['alpha']
                edited_dict['epoch'] = self.results_dict['train']['epoch']
                if not self.hparams['gamma'] is None:
                    edited_dict['gamma'] = self.hparams['gamma']
                if hasattr(self.hparams, 'auto_dim_z'):
                    edited_dict['dim_z'] = self.hparams.auto_dim_z

                for split, split_dict in self.results_dict.items():
                    for k, v in split_dict.items():
                        if not k in edited_dict.keys():
                            edited_dict[split + '_' + k] = v

                edited_dict['z_directory'] = self.hparams.folder_full_path
                edited_dict['Exp. Name'] = self.hparams.exp_name

                misc.dump_results_dict(edited_dict, results_file)
            else:
                pass
        elif test:
            # Do not log in the results.csv when the experiment is a tuning process
            if not self.hparams.tune_flag:
                results_file = os.path.join(self.hparams.result_path, 'results.csv')
                edited_dict = dict()
                edited_dict['seed'] = self.results_dict['test']['seed']
                edited_dict['tau'] = self.results_dict['test']['tau']
                edited_dict['beta'] = self.results_dict['test']['beta']
                edited_dict['alpha'] = self.results_dict['test']['alpha']
                edited_dict['epoch'] = self.results_dict['test']['epoch']
                if not self.hparams['gamma'] is None:
                    edited_dict['gamma'] = self.hparams['gamma']
                if hasattr(self.hparams, 'auto_dim_z'):
                    edited_dict['dim_z'] = self.hparams.auto_dim_z

                for split, split_dict in self.results_dict.items():
                    for k, v in split_dict.items():
                        if not k in edited_dict.keys():
                            edited_dict[split + '_' + k] = v

                edited_dict['z_directory'] = self.hparams.folder_full_path
                edited_dict['Exp. Name'] = self.hparams.exp_name

                misc.dump_results_dict(edited_dict, results_file)
            else:
                pass


    def isLastValEpoch(self):
        last_val_epoch = self.hparams.nepochs - (self.hparams.nepochs % self.hparams.check_val_every_n_epochs) - 1
        return self.current_epoch == last_val_epoch

    def configure_optimizers(self):
        optimizers_list = list()
        # This list will log which optimizers have been used in 
        # the training step of an epoch so we can know on which optimizers we need to perform scheduler step
        used_optimizers = list() 

        # Optimizer for feature extractor
        if self.feature_extractor is not None and 'feature_extractor' in self.hparams.optim_method.keys():
            params_fe = list(filter(lambda p: p.requires_grad, self.feature_extractor.parameters()))
            if len(params_fe) > 0:
                opt_fe = getattr(torch.optim, self.hparams.optim_method['feature_extractor'])(
                    params_fe, **self.hparams.optim_options['feature_extractor'])
                optimizers_list.append(opt_fe)
                used_optimizers.append(False)
            else:
                opt_fe = None
        else:
            opt_fe = None

        # Optimizer for encoder
        if self.encoder is not None and 'encoder' in self.hparams.optim_method.keys():
            params_enc = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
            if len(params_enc) > 0:
                opt_enc = getattr(torch.optim, self.hparams.optim_method['encoder'])(
                    params_enc, **self.hparams.optim_options['encoder'])
                optimizers_list.append(opt_enc)
                used_optimizers.append(False)
            else:
                opt_enc = None
        else:
            opt_enc = None

        # Optimizer for target
        if self.target is not None:
            params_tgt = list(filter(lambda p: p.requires_grad, self.target.parameters()))
            if len(params_tgt) > 0:
                opt_tgt = getattr(torch.optim, self.hparams.optim_method['target'])(
                    params_tgt, **self.hparams.optim_options['target'])
                optimizers_list.append(opt_tgt)
                used_optimizers.append(False)
            else:
                opt_tgt = None
        else:
            opt_tgt = None

        # Optimizer for adversary
        if self.adversary is not None:
            params_adv = list(filter(lambda p: p.requires_grad, self.adversary.parameters()))
            if len(params_adv) > 0:
                opt_adv = getattr(torch.optim, self.hparams.optim_method['adversary'])(
                    params_adv, **self.hparams.optim_options['adversary'])
                optimizers_list.append(opt_adv)
                used_optimizers.append(False)

        self.used_optimizers = used_optimizers

        # Scheduler for model
        if self.hparams.scheduler_method is not None:
            scheduler_list = []
            # for opt in [opt_enc, opt_tgt]:
            for opt in optimizers_list:
                if opt is not None:
                    scheduler_list.append(getattr(torch.optim.lr_scheduler, self.hparams.scheduler_method)(
                        opt, **self.hparams.scheduler_options))
            return optimizers_list, scheduler_list

        if len(optimizers_list) != 0:
            return optimizers_list
        else:
            return None
