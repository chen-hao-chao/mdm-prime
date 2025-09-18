import ray
from ray import tune
from arg_parser import get_args_parser
from eval import sample_data, eval_fid

def tuner(tuner, args):
    seed = tuner['seed']
    args.seed = seed
    sample_data(args)
    eval_fid(args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    ray.init(num_gpus=args.num_gpus)
    search_space = {"seed": tune.grid_search(args.sample_list)}
    wrapped_tuner = lambda x: tuner(x, args)
    analysis = tune.run(
        wrapped_tuner, 
        storage_path=args.storage_path,
        resources_per_trial={'cpu': 2, 'gpu': 1},
        config=search_space,
    )
