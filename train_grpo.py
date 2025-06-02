import argparse
import gc
import os
import shutil
from tqdm import tqdm
import time
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader, Sampler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from far.data import build_dataset
from far.trainers import build_trainer
from far.utils.logger_util import MessageLogger, dict2str, reduce_loss_dict, set_path_logger, setup_wandb
from far.pipelines.pipeline_far import context_cache_to_device


# From FlowGRPO: GRPO Sampler
class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每个样本重复的次数
        self.num_replicas = num_replicas  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同样本数
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 不同样本数
        self.epoch=0

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator()
            # self.epoch += 1
            g.manual_seed(self.seed + self.epoch + 1)
            # 随机选择m个不同的样本
            # print('seed', self.seed + self.epoch + 1)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            # print('indices', indices)
            # print(self.rank, 'indices', indices)
            # 每个样本重复k次，生成总样本数n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # print('re', repeated_indices)
            # 打乱顺序确保均匀分配
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # 将样本分割到各个卡
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print('per', per_card_samples)
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # 返回当前卡的样本索引
            # for idx in per_card_samples[self.rank]:
            #     yield idx
            yield per_card_samples[self.rank]
            # if self.epoch <= 1:
            #     time.sleep(2)
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态


def samples_aggregating(accelerator, samples):
    gathered_samples = []
    rewards_mean = 0
    rewards_std = 0
    
    for batch_samples in samples:
        # Gather rewards and prompt_ids from all devices
        rewards = accelerator.gather(batch_samples['rewards'])  # Shape: (total_batch_size, 1)
        prompt_ids = accelerator.gather(batch_samples['prompt_ids'])  # Shape: (total_batch_size, seq_len)
        
        # Create unique identifier for each prompt by hashing the prompt_ids
        prompt_hashes = torch.sum(prompt_ids, dim=1)  # Shape: (total_batch_size,)
        
        # Round to handle minor float differences (within 1e-4)
        # prompt_hashes = torch.round(prompt_hashes * 1e4) / 1e4
        
        # Get unique prompts and their counts
        unique_prompts, inverse_indices, prompt_counts = torch.unique(
            prompt_hashes, 
            return_inverse=True, 
            return_counts=True
        )
        
        # Initialize tensor to store advantages
        advantages = torch.zeros_like(rewards)
        
        # Compute advantages for each group separately
        # print("prompt_ids", prompt_ids)
        print('num unique prompts', len(unique_prompts))
        for i in range(len(unique_prompts)):
            mask = (prompt_hashes == unique_prompts[i])
            group_rewards = rewards[mask]
            
            # Only compute advantages if we have enough samples
            if len(group_rewards) > 1:
                mean_reward = group_rewards.mean()
                # Use a larger epsilon and handle potential numerical instability
                std_reward = torch.max(group_rewards.std(), torch.tensor(1e-4, device=rewards.device))
                group_advantages = (group_rewards - mean_reward) / std_reward
            else:
                # For single-sample groups, set advantage to 0 or some default value
                group_advantages = torch.zeros_like(group_rewards)
            
            advantages[mask] = group_advantages
            
            # Update global statistics
            rewards_mean += group_rewards.mean()
            rewards_std += std_reward
        
        # Check for NaN values before proceeding
        if torch.isnan(advantages).any():
            print(f"Warning: NaN values detected in advantages! Group sizes: {prompt_counts.tolist()}")
            advantages = torch.nan_to_num(advantages, nan=0.0)
        
        # print("advantages.shape[0], accelerator.num_processes", advantages.shape[0], accelerator.num_processes)
        # Reshape and distribute back to devices
        per_device_batch_size = advantages.shape[0] // accelerator.num_processes
        batch_samples["advantages"] = advantages[
            accelerator.process_index * per_device_batch_size:
            (accelerator.process_index + 1) * per_device_batch_size
        ].to(accelerator.device)

        # print(batch_samples['advantages'])
    
    rewards_mean /= len(samples)
    rewards_std /= len(samples)
    
    # Ensure all processes have the same rewards statistics
    rewards_mean = accelerator.gather(torch.tensor([rewards_mean], device=accelerator.device)).mean()
    rewards_std = accelerator.gather(torch.tensor([rewards_std], device=accelerator.device)).mean()
    
    return samples, rewards_mean, rewards_std


def train(args):

    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, opt, is_train=True)

    # get logger
    logger = get_logger('far', log_level='INFO')
    logger.info(accelerator.state)
    logger.info(dict2str(opt))

    # get wandb
    if accelerator.is_main_process and opt['logger'].get('use_wandb', False):
        wandb_logger = setup_wandb(name=opt['name'], save_dir=opt['path']['log'])
    else:
        wandb_logger = None

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'] + accelerator.process_index)

    # load trainer pipeline
    train_pipeline = build_trainer(opt['train']['train_pipeline'])(**opt['models'], accelerator=accelerator)

    # set optimizer
    train_opt = opt['train']
    optim_type = train_opt['optim_g'].pop('type')
    assert optim_type == 'AdamW', 'only support AdamW now'
    optimizer = torch.optim.AdamW(train_pipeline.get_params_to_optimize(train_opt['param_names_to_optimize']), **train_opt['optim_g'])
    
    # Get the training dataset
    trainset_cfg = opt['datasets']['train']
    train_dataset = build_dataset(trainset_cfg)
    # Initialize GRPO Sampler
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=trainset_cfg['batch_size_per_gpu'],
        k=trainset_cfg['num_rollouts_per_prompt'],  # Group size
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )
    # No need for shuffle, controlled by sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler, num_workers=1, pin_memory=True)

    if opt['datasets'].get('sample'):
        sampleset_cfg = opt['datasets']['sample']
        sample_dataset = build_dataset(sampleset_cfg)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sampleset_cfg['batch_size_per_gpu'], shuffle=False)
    else:
        sample_dataloader = None

    # Prepare learning rate scheduler in accelerate config
    total_batch_size = opt['datasets']['train']['batch_size_per_gpu'] * accelerator.num_processes

    num_training_steps = total_iter = opt['train']['total_iter']
    num_warmup_steps = opt['train']['warmup_iter']

    if opt['train']['lr_scheduler'] == 'constant_with_warmup':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        )
    elif opt['train']['lr_scheduler'] == 'cosine_with_warmup':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps * accelerator.num_processes,
        )
    else:
        raise NotImplementedError

    # Prepare everything with our `accelerator`.
    train_pipeline.model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        train_pipeline.model, optimizer, train_dataloader, sample_dataloader, lr_scheduler)

    # set ema after prepare everything: sync the model init weight in ema
    train_pipeline.set_ema_model(ema_decay=opt['train'].get('ema_decay'))

    # Train!
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f"  Instantaneous batch size per device = {opt['datasets']['train']['batch_size_per_gpu']}")
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Total optimization steps = {total_iter}')

    if opt['path'].get('pretrain_network', None):
        load_path = opt['path'].get('pretrain_network')
    else:
        load_path = opt['path']['models']

    global_step = resume_checkpoint(args, accelerator, load_path, train_pipeline)

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    # train_data_yielder = make_data_yielder(train_dataloader)

    msg_logger = MessageLogger(opt, global_step)
    train_iter = iter(train_dataloader)

    # while global_step < total_iter:
    for epoch in range(total_iter):
        # batch = next(train_data_yielder)
        """************************* start of an iteration*******************************"""
        samples, samples_args, reward_dict = train_pipeline.sample_grpo(accelerator, train_sampler, train_iter, opt, epoch, global_step=global_step)
        if global_step <= 1:
            global_step += 1
            continue
        # if global_step <= 3:
        #     continue
        # loss_dict = train_pipeline.train_step(batch, iters=global_step)
        
        # samples processing
        samples, rewards_mean, rewards_std = samples_aggregating(accelerator, samples)
        # print('prompt_ids', samples[0]['prompt_ids'].mean(-1))
        # print('samples[0][rewards]', samples[0]['rewards'])
        # print('samples[0][advantages]', samples[0]['advantages'])
        # log_dict = {'rewards_mean': rewards_mean, 'rewards_std': rewards_std, 'lpips': reward_dict['lpips'].mean().item(), 'ssim': reward_dict['ssim'].mean().item()}
        log_dict = {'rewards_mean': rewards_mean, 'rewards_std': rewards_std, 'mse': reward_dict['mse'].mean().item()}
        # msg_logger(log_dict)
        # print(log_dict)
        # print(reward_dict['mse'].shape)
        if wandb_logger:
            wandb_logger.log(log_dict, step=global_step)

        train_pipeline.vae.eval()
        train_pipeline.model.train()
        
        for i, sample in tqdm(
            list(enumerate(samples)),
            desc=f"Epoch {epoch}: training",
            position=0,
            disable=not accelerator.is_local_main_process,
        ):
            # Zero gradients at the start of each sample
            optimizer.zero_grad()
            
            samples_arg = samples_args[i]
            for j in range(sample['timesteps'].shape[1]): # how many steps
                # training pass
                if j == 0:
                    context_cache = {'is_cache_step': True, 'kv_cache': None, 'cached_seqlen': 0, 'multi_level_cache_init': False}
                    context_cache['is_cache_step'] = True
                else:
                    context_cache = {'is_cache_step': False, 'kv_cache': None, 'cached_seqlen': 0, 'multi_level_cache_init': False}
                    
                loss_dict, context_cache, ratio_mean, clip_frac = train_pipeline.train_step_grpo(sample, samples_arg, j, context_cache, opt, accelerator)
                ratio_dict = {'ratio_mean': ratio_mean.item(), 'clip_frac': clip_frac.item()}
                
                # Add diagnostic prints
                # print("Before backward:")
                # print(f"total_loss: {loss_dict['total_loss'].item()}")
                # print(f"requires_grad: {loss_dict['total_loss'].requires_grad}")
                # print(f"loss is nan: {torch.isnan(loss_dict['total_loss']).any()}")
                # print(f"loss is inf: {torch.isinf(loss_dict['total_loss']).any()}")
                
                accelerator.backward(loss_dict['total_loss'])
                if global_step % opt['logger']['print_freq'] == 0:
                    print('grads', list(train_pipeline.model.named_parameters())[79][1].grad.view(-1)[0].item(), list(train_pipeline.model.named_parameters())[179][1].grad.view(-1)[0].item(), list(train_pipeline.model.named_parameters())[279][1].grad.view(-1)[0].item())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(train_pipeline.model.parameters(), opt['train']['max_grad_norm'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Synchronize across all processes
                accelerator.wait_for_everyone()

                """************************* end of an iteration*******************************"""

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:

                    if train_pipeline.ema is not None:
                        train_pipeline.ema.step(accelerator.unwrap_model(train_pipeline.model))

                    global_step += 1

                    if global_step % opt['logger']['print_freq'] == 0:

                        print(ratio_dict)
                        # print('log_dict', loss_dict)
                        log_dict = reduce_loss_dict(accelerator, loss_dict)
                        log_vars = {'iter': global_step}
                        log_vars.update({'lrs': lr_scheduler.get_last_lr()})
                        log_vars.update(log_dict)
                        
                        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        log_vars['peak_mem (MB)'] = round(peak_mem, 2)
                        
                        msg_logger(log_vars)

                        if accelerator.is_main_process and wandb_logger:
                            wandb_log_dict = {
                                f'train/{k}': v
                                for k, v in log_vars.items()
                            }
                            wandb_log_dict['train/ratio_mean'] = ratio_dict['ratio_mean']
                            wandb_log_dict['train/clip_frac'] = ratio_dict['clip_frac']
                            wandb_log_dict['train/lrs'] = lr_scheduler.get_last_lr()[0]
                            wandb_logger.log(wandb_log_dict, step=global_step)

                    if global_step % opt['val']['val_freq'] == 0 or global_step == total_iter or (global_step in {2, 3} and opt['val']['eval_on_start']):

                        if sample_dataloader is not None:
                            train_pipeline.sample(sample_dataloader, opt, num_samples=2, wandb_logger=wandb_logger, global_step=global_step)

                        accelerator.wait_for_everyone()

                        if accelerator.is_main_process and 'eval_cfg' in opt['val']:
                            result_dict = train_pipeline.eval_performance(opt, global_step=global_step)
                            logger.info(result_dict)

                            if wandb_logger:
                                wandb_log_dict = {
                                    f'eval/{k}': v
                                    for k, v in result_dict.items()
                                }
                                wandb_logger.log(wandb_log_dict, step=global_step)

                        accelerator.wait_for_everyone()
                        gc.collect()
                        torch.cuda.empty_cache()

                    if accelerator.is_main_process and (global_step % opt['logger']['save_checkpoint_freq'] == 0 or global_step == total_iter):
                        save_checkpoint(args, logger, accelerator, train_pipeline, global_step, opt['path']['models'])


def resume_checkpoint(args, accelerator, output_dir, train_pipeline):
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split('-')[1])

            if train_pipeline.ema is not None:
                accelerator.print(f'Resuming ema from checkpoint {path}')
                ema_state = torch.load(os.path.join(output_dir, path, 'ema.pth'), map_location='cpu', weights_only=True)
                train_pipeline.ema.load_state_dict(ema_state)

    return global_step


def save_checkpoint(args, logger, accelerator, train_pipeline, global_step, output_dir):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints')
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f'checkpoint-{global_step}')
    accelerator.save_state(save_path)
    logger.info(f'Saved state to {save_path}')

    if train_pipeline.ema is not None:
        torch.save(train_pipeline.ema.state_dict(), os.path.join(save_path, 'ema.pth'))
        logger.info(f'Saved ema model to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default='latest')
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, help=('Max number of checkpoints to store.'))
    args = parser.parse_args()

    train(args)
