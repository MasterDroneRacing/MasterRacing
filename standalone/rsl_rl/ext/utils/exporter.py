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
# *  Data: 2025/04/25     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
import os
import torch
from standalone.rsl_rl.ext.modules import VisionActorCritic
import copy
def export_vision_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: object | None = None,
    filename="vision_policy.onnx",
    verbose=False,
    image_shape=(72, 96),
    state_shape = (24,),
    use_auxiliary_head=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _VisionOnnxPolicyExporter(policy, normalizer, image_shape, state_shape, use_auxiliary_head, verbose)
    policy_exporter.export(path, filename)

class _VisionOnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy: VisionActorCritic, normalizer=None, image_shape=(72, 96), state_shape=(24,), use_auxiliary_head=False, verbose=False):
        super().__init__()
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.verbose = verbose
        self.stem = copy.deepcopy(policy.stem)
        self.state_enc = copy.deepcopy(policy.state_enc)
        self.activation = copy.deepcopy(policy.activation)
        self.actor = copy.deepcopy(policy.actor)
        if use_auxiliary_head:
            self.auxiliary_head = copy.deepcopy(policy.aux_decoder)

        self.is_recurrent = policy.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(policy.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, state, depth_img, h_in, c_in):
        '''
        img shape: (batch_size, channels, height, width)
        state shape: (batch_size, state_dim)
        '''
        img_h = depth_img.shape[2]
        img_w = depth_img.shape[3]
        concat_state = torch.cat([state, depth_img.view(depth_img.shape[0], -1)], dim=1)
        x_in = self.normalizer(concat_state)
        _depth_img = x_in[..., -img_h * img_w :].view(-1, 1, img_h, img_w)
        _state = x_in[..., :-img_h * img_w]
        feat = self.activation(self.stem(_depth_img) + self.state_enc(_state))
        x, (h, c) = self.rnn(feat.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c
    
    def forward(self, state, depth_image):
        '''
        img shape: (batch_size, channels, height, width)
        state shape: (batch_size, state_dim)
        '''
        img_h = depth_image.shape[2]
        img_w = depth_image.shape[3]
        concat_state = torch.cat([state, depth_image.view(depth_image.shape[0], -1)], dim=1)
        x_in = self.normalizer(concat_state)
        _depth_img = x_in[..., -img_h * img_w :].view(-1, 1, img_h, img_w)
        _state = x_in[..., :-img_h * img_w]
        feat = self.activation(self.stem(_depth_img) + self.state_enc(_state))
        if hasattr(self, 'auxiliary_head'):
            aux_feat = self.auxiliary_head(feat)
            return self.actor(feat), aux_feat.sigmoid()
        else:
            return self.actor(feat)
    
    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            img = torch.zeros(1, 1, *self.image_shape)
            state = torch.zeros(1, *self.state_shape)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(state, img, h_in, c_in)
            torch.onnx.export(
                self,
                (state, img, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["state", "img", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )

        else:
            img = torch.zeros(1, 1, *self.image_shape)
            state = torch.zeros(1, *self.state_shape)
            if hasattr(self, 'auxiliary_head'):
                actions, aux_actions = self(state, img)
                torch.onnx.export(
                    self,
                    (state, img),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["state", "img"],
                    output_names=["actions", "auxiliary"],
                    dynamic_axes={},
                )
            else:
                actions = self(state, img)
                torch.onnx.export(
                    self,
                    (state, img),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["state", "img"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
