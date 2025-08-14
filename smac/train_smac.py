from sys import argv

from smac.environment import get_smacv2_obs_shape
from smac.training_config import ExperimentSMACv2Config, SMACv2Config
from smac.training_utils import create_sf_config_smacv2, run_smacv2

def recursive_update(experiment: dict, key, value):
    if key in experiment:
        experiment[key] = value
        return True
    else:
        for k, v in experiment.items():
            if isinstance(v, dict):
                if recursive_update(v, key, value):
                    return True
        return False

def update_dict(target_dict, keys, values):
    for key, value in zip(keys, values):
        if recursive_update(target_dict, key, value):
            print(f'Updated {key} to {value}')
        else:
            raise KeyError(f'Could not find {key} in experiment')


def parse_args_to_items(argv_):
    keys = []
    values = []

    for arg in argv_[1:]:
        key, value = arg.split('=')
        key = key.replace('--', '')

        keys.append(key)
        values.append(value)

    return keys, values


def main():
    experiment = ExperimentSMACv2Config()
    experiment = create_sf_config_smacv2(experiment).__dict__
    keys, values = parse_args_to_items(list(argv))
    update_dict(experiment, keys, values)

    if 'map_config' in keys:
        import yaml
        with open(values[keys.index('map_config')], 'r') as f:
            cfg = yaml.safe_load(f)['env_args']
        experiment['environment']['env_extra_config'] |= cfg

    # obs_shape = 82
    obs_shape = get_smacv2_obs_shape(SMACv2Config(**experiment['environment']['env_extra_config']))

    experiment['environment']['obs_shape'] = obs_shape
    experiment['environment']['grid_config']['num_agents'] = \
        experiment['environment']['env_extra_config']['capability_config']['n_units']
    run_smacv2(config=experiment)


if __name__ == '__main__':
    main()
