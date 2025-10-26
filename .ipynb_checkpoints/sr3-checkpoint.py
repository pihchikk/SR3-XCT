import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import psutil
from GPUtil import getGPUs
import yaml

####### GPU-CPU usage monitoring

def get_cpu_usage():
    return psutil.cpu_percent()

def get_gpu_usage():
    gpus = getGPUs()
    gpu_usage = [gpu.load * 100 for gpu in gpus]
    return gpu_usage

def load_config(config_path: str):
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

####### PSNR-SSIM-controlled evaluation 

def evaluate_diffusion_model(diffusion, val_loader, current_step, result_path, logger, wandb_logger=None, opt=None, num_iterations=15):
    logger.info('Begin Model Evaluation.')
    idx = 0
    os.makedirs(result_path, exist_ok=True)

    for x, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)

        # Initialize variables to track best SR image and metrics
        best_sr_img = None
        best_psnr = -1.0
        best_ssim = -1.0
        best_psnr_idx = -1
        best_ssim_idx = -1

        # Initialize array to store PSNR and SSIM for each iteration
        psnr_scores = []
        ssim_scores = []

        # Generate SR images in a loop
        sr_imglist = []
        for iter in range(num_iterations):
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            sr_img = Metrics.tensor2img(visuals['SR'][-1])  # uint8

            # Compute PSNR and SSIM for current SR image compared to HR image
            current_psnr = Metrics.calculate_psnr(sr_img, hr_img)
            current_ssim = Metrics.calculate_ssim(sr_img, hr_img)

            # Log PSNR and SSIM for current iteration
            logger.info('# Iteration {} # PSNR: {:.4f}, SSIM: {:.4f}'.format(iter, current_psnr, current_ssim))

            # Store PSNR and SSIM scores for current iteration
            psnr_scores.append(current_psnr)
            ssim_scores.append(current_ssim)

            sr_imglist.append(sr_img)

            # Save intermediate results (optional)
            if opt['save_intermediate']:
                Metrics.save_img(sr_img, '{}/{}_{}_iter{}_sr.png'.format(result_path, current_step, idx, iter))

            # Log evaluation data to W&B (if enabled)
            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, sr_img, hr_img, current_psnr, current_ssim)

        # Determine the best image based on PSNR and SSIM scores
        best_index = select_best_image(psnr_scores, ssim_scores)

        # Save final best SR image
        best_sr_img = sr_imglist[best_index]  # Convert best SR tensor to image
        Metrics.save_img(best_sr_img, '{}/{}_{}_best_sr.png'.format(result_path, current_step, idx))

        # Calculate final PSNR and SSIM for the selected best image
        best_psnr = Metrics.calculate_psnr(best_sr_img, hr_img)
        best_ssim = Metrics.calculate_ssim(best_sr_img, hr_img)

        # Log final PSNR and SSIM for the selected best image
        logger.info('# Validation # Best Image PSNR: {:.4f}'.format(best_psnr))
        logger.info('# Validation # Best Image SSIM: {:.4f}'.format(best_ssim))

        # Log metrics to W&B (if enabled)
        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'Best Image PSNR': float(best_psnr),
                'Best Image SSIM': float(best_ssim)
            })

def select_best_image(psnr_scores, ssim_scores):

    psnr_ranks = np.argsort(np.argsort(psnr_scores))
    ssim_ranks = np.argsort(np.argsort(ssim_scores))

    # Combine PSNR and SSIM ranks
    combined_scores = psnr_ranks + ssim_ranks

    # Find the indices of the maximum combined score
    best_indices = np.where(combined_scores == combined_scores.max())[0]

    # Among the best indices, select the one with the highest PSNR score
    best_index = best_indices[np.argmax(np.array(psnr_scores)[best_indices])]

    return best_index
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
        default='config/inference/32_256/deep.yaml',
        help='Configuration file path (JSON or YAML)')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-n', '--num_iterations', type=int, default=1,
        help='Number of diffusion iterations for evaluation')  # Add this line
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    
    args = parser.parse_args()
    
    # Load config file
    opt = load_config(args.config)
    
    # Transfer command line arguments to config
    opt['phase'] = args.phase  # Command line args take precedence
    
    # Convert to NoneDict
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)

    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    print("Before training/validation branch phase:", opt['phase'])  # Debug print
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                if current_step >= opt['train']['ema_scheduler']['step_start_ema'] and current_step % opt['train']['ema_scheduler']['update_ema_every'] == 0:
                    diffusion.update_lr(opt['train']['ema_scheduler']['ema_decay'])
                    
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        
                        def ensure_3d(img):
                            return np.expand_dims(img, axis=2) if len(img.shape)==2 else img

                        fake_img = ensure_3d(fake_img)
                        sr_img = ensure_3d(sr_img)
                        hr_img = ensure_3d(hr_img)

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
        
    else:
        result_path = '{}'.format(opt['path']['results'])
        evaluate_diffusion_model(
            diffusion, 
            val_loader, 
            current_step, 
            result_path, 
            logger, 
            opt=opt,
            num_iterations=args.num_iterations  # Pass the argument here
        )
       


    

