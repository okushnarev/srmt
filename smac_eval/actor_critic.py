from typing import Dict, Optional

import torch
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic, ActorCriticSharedWeights
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from torch import Tensor, nn as nn


def concat_td(structures):
    concatenated = TensorDict()
    if not structures:
        return concatenated
    keys = structures[0].keys()
    for key in keys:
        tensors_to_stack = [s[key] for s in structures]
        if tensors_to_stack[0].dim() == 0:
            tensors_to_stack = [t.unsqueeze(0) for t in tensors_to_stack]
        concatenated[key] = torch.cat(tensors_to_stack, dim=0)
    return concatenated


class MultiModelActorCritic(ActorCritic):
    """
    This class acts as a container, holding a separate instance of
    ActorCriticSharedWeights for each agent.
    """

    def __init__(self, model_factory, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(self.cfg, dict):
            self.cfg = AttrDict(**self.cfg)

        self.num_agents = self.cfg.environment['grid_config']['num_agents']
        if not self.num_agents > 0:
            raise ValueError("num_agents must be a positive integer for this model.")

        self.agent_models = nn.ModuleList(
            [ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg) for _ in range(self.num_agents)]
        )

        self.encoders = [model.encoder for model in self.agent_models]
        self.cores = [model.core for model in self.agent_models]
        self.decoders = [model.decoder for model in self.agent_models]

        self.action_parameterization = self.get_action_parameterization(self.agent_models[0].decoder.get_out_size())
        self.last_action_distribution = None

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        head_outputs = []
        for idx, model in enumerate(self.agent_models):
            head_outputs.append(model.forward_head(normalized_obs_dict[idx::self.num_agents]))
        return torch.cat(head_outputs, dim=0)

    def forward_core(self,
                     head_output: Tensor,
                     rnn_states,
                     agent_memory=None,
                     global_memory=None,
                     history_seq=None,
                     action_seq=None, ):

        outputs, new_rnn_states, additional_outputs = [], [], []
        attn_condition = getattr(self.cfg, 'attn_core', None) == True
        for idx, model in enumerate(self.agent_models):
            if attn_condition:
                out, rnn, add_out = model.core(
                    head_output=head_output[idx::self.num_agents],
                    rnn_states=rnn_states[idx::self.num_agents],
                    agent_memory=agent_memory[idx::self.num_agents] if agent_memory is not None else None,
                    global_memory=global_memory[idx::self.num_agents] if global_memory is not None else None,
                    history_seq=history_seq[idx::self.num_agents] if history_seq is not None else None,
                    action_seq=action_seq[idx::self.num_agents] if action_seq is not None else None,
                )
                additional_outputs.append(add_out)
            else:
                out, rnn = model.core(
                    head_output[idx::self.num_agents],
                    rnn_states[idx::self.num_agents]
                )

            outputs.append(out)
            new_rnn_states.append(rnn)

        outputs = torch.cat(outputs, dim=0)
        new_rnn_states = torch.cat(new_rnn_states, dim=0)
        additional_outputs = concat_td(additional_outputs)

        if attn_condition:
            return outputs, new_rnn_states, additional_outputs
        else:
            return outputs, new_rnn_states

    def forward_tail(
            self,
            core_output,
            values_only: bool,
            sample_actions: bool,
            action_mask: Optional[Tensor] = None) -> TensorDict:

        decoder_outputs, values = [], []

        for idx, model in enumerate(self.agent_models):
            dec_out = model.decoder(core_output[idx::self.num_agents])
            val_out = model.critic_linear(dec_out)

            decoder_outputs.append(dec_out)
            values.append(val_out)

        decoder_outputs = torch.cat(decoder_outputs, dim=0).squeeze()
        values = torch.cat(values, dim=0).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_outputs,
                                                                                                 action_mask)

        result['action_logits'] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self,
                normalized_obs_dict,
                rnn_states,
                agent_memory=None,
                global_memory=None,
                history_seq=None,
                action_seq=None,
                values_only: bool = False,
                action_mask: Optional[Tensor] = None) -> TensorDict:

        x = self.forward_head(normalized_obs_dict)

        if getattr(self.cfg, 'attn_core', None) == True:
            x, new_rnn_states, additional_outputs = self.forward_core(x, rnn_states,
                                                                      agent_memory=agent_memory,
                                                                      global_memory=global_memory,
                                                                      history_seq=history_seq,
                                                                      action_seq=action_seq
                                                                      )
        else:
            x, new_rnn_states = self.forward_core(x, rnn_states)
            additional_outputs = {}

        agent_new_memory = additional_outputs.get('agent_new_memory', None)
        global_memory = additional_outputs.get('global_memory', None)
        new_history_seq = additional_outputs.get('new_history_seq', None)

        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)

        result["new_rnn_states"] = new_rnn_states
        if getattr(self.cfg, 'core_memory', None) == True:
            result["agent_new_memory"] = agent_new_memory
            result["global_memory"] = global_memory
        if getattr(self.cfg, 'attn_core', None) == True:
            result["new_history_seq"] = new_history_seq

        return result
