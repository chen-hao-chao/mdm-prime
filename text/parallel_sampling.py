import sys
import main
import hydra
import utils
import dataloader
import lightning as L

import ray
from ray import tune

def tuner(tuner, config, logger, tokenizer):
    split_index = tuner['split_index']
    config.sampling.split_index = split_index

    sys.path.insert(0, "/app")
    import main
    main._sample_and_save(config, logger, tokenizer)
    print("Finish sampling: {}".format(split_index))

# ====================================


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main__(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    main._print_config(config, resolve=True, save_cfg=True)
    
    ray.init(num_gpus=8)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    
    search_space = {
        "split_index": tune.grid_search([0,1,2,3,4,5,6,7]),
    }

    wrapped_tuner = lambda x: tuner(x, config, logger, tokenizer)

    analysis = tune.run(
        wrapped_tuner, 
        storage_path="/app/results_ray",
        resources_per_trial={'cpu': 16, 'gpu': 1},
        config=search_space,
    )


if __name__ == "__main__":
    main__()
