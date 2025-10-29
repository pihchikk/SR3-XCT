import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from safetensors.torch import save_file, load_file

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])

            # Add learning rate scheduler with cosine annealing
            # This gradually reduces learning rate over training for better convergence
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optG,
                T_max=opt['train']['n_iter'],
                eta_min=opt['train']["optimizer"]["lr"] * 0.01  # Minimum LR is 1% of initial
            )

            self.log_dict = OrderedDict()
        self.load_network()
        #self.print_network()

    def update_lr(self, ema_decay):
        for param_group in self.optG.param_groups:
            param_group['lr'] = ema_decay * param_group['lr']

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # Step the learning rate scheduler
        self.scheduler.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['lr'] = self.optG.param_groups[0]['lr']  # Log current learning rate

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                ('obtatining self.netG.super_resolution')
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        network = self.netG
        if isinstance(self.netG, torch.nn.DataParallel):
            network = network.module
    
        # Generator
        gen_state = {k: v.cpu() for k, v in network.state_dict().items()}
        gen_st_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_gen.safetensors')
        save_file(gen_state, gen_st_path)
        logger.info(f'Saved generator (safetensors) in [{gen_st_path}]')
    
        # Optimizer
        opt_state = self.optG.state_dict()
        opt_st_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_opt.safetensors')
        save_file(opt_state, opt_st_path)
        logger.info(f'Saved optimizer (safetensors) in [{opt_st_path}]')

        # Scheduler
        scheduler_state = self.scheduler.state_dict()
        scheduler_st_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_scheduler.safetensors')
        save_file(scheduler_state, scheduler_st_path)
        logger.info(f'Saved scheduler (safetensors) in [{scheduler_st_path}]')

        # Optional .pth backup
        gen_pth = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_gen.pth')
        torch.save(gen_state, gen_pth)
        opt_pth = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_opt.pth')
        torch.save({'epoch': epoch, 'iter': iter_step, 'optimizer': opt_state, 'scheduler': scheduler_state}, opt_pth)
    
    def load_network(self):
        load_base = self.opt['path']['resume_state']  # e.g., 'pretrained models/32_256'
        network = self.netG
        if isinstance(self.netG, torch.nn.DataParallel):
            network = network.module
    
        # Generator
        gen_st = f'{load_base}_gen.safetensors'
        gen_pth = f'{load_base}_gen.pth'
        if os.path.exists(gen_st):
            logger.info(f'Loading generator from safetensors [{gen_st}]')
            gen_state = load_file(gen_st)
        elif os.path.exists(gen_pth):
            logger.info(f'Loading generator from pth [{gen_pth}]')
            gen_state = torch.load(gen_pth, map_location='cpu')
        else:
            raise FileNotFoundError(f'No generator file found ({gen_st} or {gen_pth})')
        network.load_state_dict(gen_state, strict=(not self.opt['model']['finetune_norm']))
    
        if self.opt['phase'] == 'train':
            # Optimizer
            opt_st = f'{load_base}_opt.safetensors'
            opt_pth = f'{load_base}_opt.pth'
            if os.path.exists(opt_st):
                logger.info(f'Loading optimizer from safetensors [{opt_st}]')
                opt_state = load_file(opt_st)
                self.begin_epoch = int(opt_state.get('epoch', 0).item()) if 'epoch' in opt_state else 0
                self.begin_step = int(opt_state.get('iter', 0).item()) if 'iter' in opt_state else 0
            elif os.path.exists(opt_pth):
                logger.info(f'Loading optimizer from pth [{opt_pth}]')
                tmp = torch.load(opt_pth, map_location='cpu')
                opt_state = tmp['optimizer']
                self.begin_epoch = tmp.get('epoch', 0)
                self.begin_step = tmp.get('iter', 0)
            else:
                raise FileNotFoundError(f'No optimizer file found ({opt_st} or {opt_pth})')

            self.optG.load_state_dict(opt_state)

            # Scheduler
            scheduler_st = f'{load_base}_scheduler.safetensors'
            scheduler_pth = f'{load_base}_scheduler.pth'
            if os.path.exists(scheduler_st):
                logger.info(f'Loading scheduler from safetensors [{scheduler_st}]')
                scheduler_state = load_file(scheduler_st)
                self.scheduler.load_state_dict(scheduler_state)
            elif os.path.exists(opt_pth):
                # Try to load from pth if safetensors doesn't exist
                tmp = torch.load(opt_pth, map_location='cpu')
                if 'scheduler' in tmp:
                    logger.info(f'Loading scheduler from pth [{opt_pth}]')
                    self.scheduler.load_state_dict(tmp['scheduler'])
            # If scheduler state doesn't exist, just use fresh scheduler (backward compatibility)