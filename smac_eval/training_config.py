from typing import Literal, Optional

from pydantic import BaseModel, root_validator, validator

from smac_eval.encoder import FCNNEncoderConfig
from srmt.model import CoreConfig


class SMACv2Config(BaseModel):
    continuing_episode = False
    difficulty = '7'
    game_version: Optional[str] = None
    map_name = '10gen_terran'
    move_amount = 2
    obs_all_health = True
    obs_instead_of_state = False
    obs_last_action = False
    obs_own_health = True
    obs_pathing_grid = False
    obs_terrain_height = False
    obs_timestep_number = False
    reward_death_value = 10
    reward_defeat = 0
    reward_negative_scale = 0.5
    reward_only_positive = True
    reward_scale = True
    reward_scale_rate = 20
    reward_sparse = False
    reward_win = 200
    replay_dir = ''
    replay_prefix = ''
    conic_fov = False
    obs_own_pos = True
    use_unit_ranges = True
    min_attack_range = 2
    num_fov_actions = 12
    capability_config = {
        'n_units':         5,
        'n_enemies':       5,
        'team_gen':        {
            'dist_type':            'weighted_teams',
            'unit_types':           ['marine', 'marauder', 'medivac'],
            'exception_unit_types': ['medivac'],
            'weights':              [0.45, 0.45, 0.1],
            'observe':              True,
        },
        'start_positions': {
            'dist_type': 'surrounded_and_reflect',
            'p':         0.5,
            'n_enemies': 5,
            'map_x':     32,
            'map_y':     32,
        },
    }
    state_last_action = True
    state_timestep_number = False
    step_mul = 8
    heuristic_ai = False
    debug = True
    prob_obs_enemy = 1.0
    action_mask = True


class EnvironmentSMACv2Config(BaseModel):
    env: Literal['SMAC-v2'] = 'SMAC-v2'
    map_config: str = 'sc2_gen_terran.yaml'
    env_extra_config: SMACv2Config = SMACv2Config()
    grid_config: dict = dict(num_agents=None)
    # num_agents: int = 5
    agent_bins: Optional[list] = None
    obs_shape: int = None
    worker_index: int = None
    vector_index: int = None
    env_id: int = None


class ExperimentSMACv2Config(BaseModel):
    environment: EnvironmentSMACv2Config = EnvironmentSMACv2Config()
    encoder: FCNNEncoderConfig = FCNNEncoderConfig(
        num_outputs=256,  # Must be equal core_hidden_size
        hidden_layers=[],
        dropout=0.2,
    )

    core: CoreConfig = CoreConfig(
        core_hidden_size=256,
        num_attention_heads=4,
        max_position_embeddings=16384,
    )

    attn_core: bool = True
    core_memory: bool = True
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    rollout: int = 10
    num_workers: int = 2
    num_envs_per_worker: int = 2
    worker_num_splits: int = 1
    max_policy_lag: int = 1

    recurrence: int = 1
    rnn_size: int = 256
    use_rnn: bool = False

    ppo_clip_ratio: float = 0.2

    exploration_loss_coeff: float = 0.03
    learning_rate: float = 0.0002
    gamma: float = 0.9716
    batch_size: int = 9000

    force_envs_single_thread: bool = True
    optimizer: Literal['adam', 'lamb'] = 'adam'
    restart_behavior: str = 'overwrite'
    normalize_returns: bool = False
    async_rl: bool = False

    num_batches_per_epoch: int = 10
    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = 'win_rate'
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 60

    keep_checkpoints: int = 5
    stats_avg: int = 10
    train_for_env_steps: int = 15_000_000

    pbt_mix_policies_in_one_env: bool = False
    num_policies: int = 1

    seed: Optional[int] = 42

    adaptive_stddev: Optional[bool] = True
    mode: Literal['PPO', 'IPPO'] = 'PPO'

    lr_schedule: str = 'kl_adaptive_minibatch'
    lr_adaptive_min: float = 1e-6

    experiment: str = 'exp_smacv2'
    train_dir: str = 'experiments/train_dir'

    use_wandb: bool = False

    env: Literal['SMAC-v2'] = 'SMAC-v2'
    serial_mode: bool = False

    @root_validator
    def ensure_num_outputs_matches_core(cls, values):
        encoder_cfg = values['encoder']
        core_cfg = values['core']

        if encoder_cfg.num_outputs != core_cfg.core_hidden_size:
            encoder_cfg.num_outputs = core_cfg.core_hidden_size
        return values
