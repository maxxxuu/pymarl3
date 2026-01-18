# Multi-Agent Proximal Policy Optimization (MAPPO)
# Adapted from https://github.com/marlbenchmark/on-policy
import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class MAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if getattr(self.args, 'common_reward', False) else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if getattr(self.args, 'common_reward', False):
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "MAPPO Learner: mask.sum() == 0 at t_env {}".format(t_env)
            )
            return

        mask = mask.repeat(1, 1, self.n_agents)

        # Forward pass to get policy outputs
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # (batch_size, time, n_agents, n_actions)

        pi = mac_out

        # Get values for GAE computation
        with th.no_grad():
            v_pred = self.target_critic(batch)
            v_pred = v_pred[:, :-1].squeeze(3)  # (batch_size, time, n_agents)

        if self.args.standardise_returns:
            v_pred_normalized = (v_pred - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        else:
            v_pred_normalized = v_pred

        # Compute advantages using GAE
        advantages = self._compute_gae(
            rewards, v_pred_normalized, terminated, mask
        )

        # Compute returns
        returns = advantages + v_pred_normalized

        if self.args.standardise_returns:
            self.ret_ms.update(returns)
            advantages = (returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        advantages = advantages.detach()
        returns = returns.detach()

        # PPO policy update with clipped objective
        pi[mask == 0] = 1.0
        # actions already has shape (batch, time, n_agents, 1); gather over the action dim
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)

        # Entropy regularization
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

        # PPO clipped objective
        ratio = th.exp(log_pi_taken)
        clipped_ratio = th.clamp(ratio, 1.0 - getattr(self.args, 'ppo_clip_ratio', 0.2), 1.0 + getattr(self.args, 'ppo_clip_ratio', 0.2))
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        pg_loss = -th.min(surr1, surr2)

        pg_loss = (
            (pg_loss + getattr(self.args, 'entropy_coef', 0.01) * entropy) * mask
        ).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip
        )
        self.agent_optimiser.step()

        # Critic update
        critic_loss = self._update_critic(batch, returns, mask)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss, t_env)
            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def _compute_gae(self, rewards, v_pred, terminated, mask, gamma=None, gae_lambda=None):
        """Compute Generalized Advantage Estimation"""
        if gamma is None:
            gamma = self.args.gamma
        if gae_lambda is None:
            gae_lambda = getattr(self.args, 'gae_lambda', 0.95)

        batch_size = rewards.size(0)
        max_t = rewards.size(1)
        n_agents = rewards.size(2)

        advantages = th.zeros_like(rewards)
        gae = th.zeros_like(rewards[:, 0])

        # Compute GAE backwards through time
        for t in reversed(range(max_t)):
            if t == max_t - 1:
                next_v = th.zeros_like(v_pred[:, t])
            else:
                next_v = v_pred[:, t + 1]

            td_error = rewards[:, t] + gamma * next_v * (1 - terminated[:, t]) - v_pred[:, t]
            gae = td_error + gamma * gae_lambda * gae * (1 - terminated[:, t])
            advantages[:, t] = gae

        return advantages

    def _update_critic(self, batch, returns, mask):
        """Update value function critic"""
        v_pred = self.critic(batch)[:, :-1].squeeze(3)

        td_error = returns.detach() - v_pred
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        return loss.item()

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
