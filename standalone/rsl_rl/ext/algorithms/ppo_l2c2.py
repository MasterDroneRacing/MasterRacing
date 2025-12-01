import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from standalone.rsl_rl.ext.storage import RolloutStorageL2C2

class PPOL2C2:
    policy: ActorCritic
    def __init__(self,
                 policy,
                 env,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                value_smoothness_coef=0.1,
                smoothness_upper_bound=1.0,
                smoothness_lower_bound=0.1,
                normalize_advantage_per_mini_batch=False,
                # RND parameters
                rnd_cfg: dict | None = None,
                # Symmetry parameters
                symmetry_cfg: dict | None = None,
                # Distributed training parameters
                multi_gpu_cfg: dict | None = None,
                 ):
        self.env = env
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageL2C2.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageL2C2(
            training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.policy.test()
    
    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        if torch.norm(self.transition.observations).mean() > 1e-4:
            self.storage.add_transitions(self.transition)

        self.transition.clear()
        self.policy.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_smooth_loss = 0
        if self.policy.is_recurrent:
            raise NotImplementedError()
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch, 
            critic_obs_batch, 
            next_obs_batch, 
            cont_batch, 
            actions_batch, 
            target_values_batch, 
            advantages_batch, 
            returns_batch, 
            old_actions_log_prob_batch,
            old_mu_batch, 
            old_sigma_batch, 
            hid_states_batch, 
            masks_batch,
            ) in generator:
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) 
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) 
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5, 
                            axis=-1,
                            )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Smooth loss
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon
                value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)

                policy_smooth_loss = torch.square(torch.norm(mu_batch - self.policy.act_inference(mix_obs_batch)[0], dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(value_batch - self.policy.evaluate(mix_obs_batch), dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss
                with torch.inference_mode():
                    action_smoothness = torch.norm(mu_batch - self.policy.act_inference(next_obs_batch)[0], dim=-1).mean()
                    
                loss += smooth_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_smooth_loss += smooth_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_smooth_loss /= num_updates
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "smooth_loss": mean_smooth_loss
        }