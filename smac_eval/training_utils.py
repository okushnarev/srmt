from argparse import Namespace

import wandb
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import make_runner
from sample_factory.utils.utils import log

from smac_eval.environment import register_smacv2_env
from smac_eval.register_training_utils import register_custom_actor_critic, register_custom_encoder, \
    register_msg_handlers
from smac_eval.training_config import ExperimentSMACv2Config
from srmt.register_training_utils import register_custom_core


def create_sf_config_smacv2(exp: ExperimentSMACv2Config):
    custom_argv = [f'--env={exp.env}']
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    parser.set_defaults(**exp.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    return final_cfg


def run_smacv2(config=None):
    register_custom_encoder()

    exp = ExperimentSMACv2Config(**config)
    flat_config = Namespace(**exp.dict())
    env_name = exp.environment.env
    register_smacv2_env(env_name)

    log.debug(f'env_name = {env_name}')
    log.info(flat_config)

    if str(config.get('attn_core', None)).lower() == 'true':
        register_custom_core()  # Attention Core

    if exp.use_wandb:
        wandb_run_name = exp.experiment
        wandb.init(
            project='srmt',
            config=exp.dict(),
            save_code=False,
            sync_tensorboard=True,
            anonymous='allow',
            job_type=exp.environment.env,
            group='train',
            name=wandb_run_name
        )

    sfconfig = create_sf_config_smacv2(exp)

    flat_config, runner = make_runner(sfconfig)
    register_msg_handlers(flat_config, runner)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status


def run_smacv2_IPPO(config):
    register_custom_encoder()

    exp = ExperimentSMACv2Config(**config)
    flat_config = Namespace(**exp.dict())

    env_name = exp.environment.env
    register_smacv2_env(env_name)

    log.debug(f'env_name = {env_name}')
    log.info(flat_config)

    if str(config.get('attn_core', None)).lower() == 'true':
        register_custom_core()  # Attention Core
        register_custom_actor_critic()

    sfconfig = create_sf_config_smacv2(exp)

    flat_config, runner = make_runner(sfconfig)
    register_msg_handlers(flat_config, runner)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status
