# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/03/28     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************

import torch.nn as nn
import torch.optim as optim

from standalone.rsl_rl.ext.modules import StudentTeacher
from standalone.rsl_rl.ext.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""
    policy: StudentTeacher

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        **kwargs
    ):
        if kwargs:
            print(
                "Distillation.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.device = device
        self.learning_rate = learning_rate

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None # initialize later
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorage.Transition()

        # distillation hyperparameters
        self.num_learning_epochs = num_learning_epochs

        self.num_updates = 0


    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        action_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            action_shape,
            self.device
        )

    def act(self, obs, teacher_obs):
        # compute actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        # record rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        # clear the transition
        self.transition.clear()
        self.policy.reset(dones)

    def update(self,):
        self.num_updates += 1
        mean_behaviour_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset()
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, in self.storage.generator():
                # inference the student for gradient calculation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = nn.functional.mse_loss(actions, privileged_actions)

                # total loss
                loss = loss + behavior_loss
                mean_behaviour_loss += behavior_loss.item()

                cnt += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.policy.detach_hidden_states()
            loss = 0

        mean_behaviour_loss /= cnt
        self.storage.clear()
        self.policy.reset()

        # construct the loss dictionary
        return {"behavior": mean_behaviour_loss}



        