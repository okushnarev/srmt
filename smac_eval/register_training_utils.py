import numpy as np
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import ActorCritic, obs_space_without_action_mask
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace, PolicyID
from sample_factory.utils.utils import log
from tensorboardX import SummaryWriter

from smac_eval.actor_critic import MultiModelActorCritic
from smac_eval.encoder import FCNNEncoder


def register_msg_handlers(cfg: Config, runner: Runner):
    runner.register_episodic_stats_handler(smacv2_extra_episodic_stats_processing)
    runner.register_observer(CustomExtraSummariesObserver())


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return FCNNEncoder(cfg, obs_space)


def register_custom_encoder():
    global_model_factory().register_encoder_factory(make_custom_encoder)


def smacv2_extra_episodic_stats_processing(runner: Runner, msg: dict, policy_id: PolicyID) -> None:
    pass


def smacv2_extra_summaries(runner: Runner, policy_id: PolicyID, summary_writer: SummaryWriter, env_steps: int):
    group = f'smacv2_custom/{policy_id}'
    policy_avg_stats = runner.policy_avg_stats
    for key in policy_avg_stats:
        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(f'{group}/{key}', avg, env_steps, display_name=f'{policy_id}/{key}')
        log.debug(f'{policy_id}-{key}: {float(avg):.3f}')


class CustomExtraSummariesObserver(AlgoObserver):
    def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int) -> None:
        smacv2_extra_summaries(runner, policy_id, writer, env_steps)


def make_custom_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)

    if cfg.actor_critic_share_weights:
        return MultiModelActorCritic(model_factory, obs_space, action_space, cfg)
    else:
        raise NotImplementedError(f'Actor critic separate weights not implemented.')


def register_custom_actor_critic():
    global_model_factory().register_actor_critic_factory(make_custom_actor_critic)
