from typing import Dict, List, Literal

import torch
from pydantic import BaseModel, validator
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from torch import nn as nn


class FCNNEncoderConfig(BaseModel):
    obs_key: str = 'obs'  # in case the observations carry other keys
    num_inputs: int = 82
    hidden_layers: List[int] = [10] * 3
    num_outputs: int = 16
    activation: Literal['relu', 'tanh', 'elu', 'gelu', 'silu', 'leaky_relu'] = 'relu'
    layer_norm: bool = False
    dropout: float = 0.0

    @validator('hidden_layers')
    def _nonempty_and_positive(cls, v: List[int]) -> List[int]:
        if any(h <= 0 for h in v):
            raise ValueError('hidden_layers must all be > 0')
        return v

    @validator('dropout')
    def _valid_dropout(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError('dropout must be in [0.0, 1.0)')
        return v


class FCNNEncoder(Encoder):
    """
    A simple fully-connected encoder for Sample Factory.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace = None):
        super().__init__(cfg)

        self.cfg = FCNNEncoderConfig(**cfg.encoder)
        self.cfg.num_inputs = obs_space['obs'].shape[0]
        self.activation_layer = self._get_activation_layer(self.cfg.activation)
        self.model = self._build_model()

    def _get_activation_layer(self, name: str) -> nn.Module:
        activations = {
            'relu':       nn.ReLU(),
            'tanh':       nn.Tanh(),
            'elu':        nn.ELU(),
            'gelu':       nn.GELU(),
            'silu':       nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        }
        assert name in activations, f'Activation layer {name} not in {activations.keys()}'
        return activations[name]

    def _build_model(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        input_dim = self.cfg.num_inputs

        for hidden_dim in self.cfg.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_layer)
            if self.cfg.dropout > 0.0:
                layers.append(nn.Dropout(self.cfg.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.cfg.num_outputs))
        return nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(observations, dict):
            observations = observations[self.cfg.obs_key]
        return self.model(observations)

    def get_out_size(self) -> int:
        return self.cfg.num_outputs

    def reset(self):
        pass
