import argparse
import logging

from allennlp.common import Params

from sitod.experiment import Experiment


def main(data_root_dir: str, experiment_root_dir: str, config: str):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    params = Params.from_file(config)
    experiment = Experiment.from_params(params)
    experiment.run_experiment(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--experiment_root_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(data_root_dir=args.data_root_dir, experiment_root_dir=args.experiment_root_dir, config=args.config)
