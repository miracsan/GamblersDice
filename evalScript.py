import argparse
import os

from constructModel import load_model, get_num_pool_ops
from dataloaders import construct_dataloaders
from evalCnnSegmentation import eval_cnn
from utils import get_configs_from_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="softmax_response, mc_dropout, extra_class",
        default="softmax_response",
    )
    parser.add_argument("--save_uncs", type=int, default=0)
    parser.add_argument(
        "--alias",
        help="the alias of the experiment that we want to evaluate",
        default="dice_covid19",
    )
    parser.add_argument(
        "--dataset",
        help="covid19, whs, spleen, thor",
        default="covid19",
    )
    parser.add_argument(
        "--checkpoint",
        help="name of the checkpoint under the alias folder that we want to use for evaluation",
        default="best_checkpoint",
    )

    options = parser.parse_args()

    model_path = os.path.join("results", options.alias, options.checkpoint)
    model, _, _, _ = load_model(model_path)

    _, batch_size, _ = get_configs_from_dataset(options.dataset)
    num_pool_ops = get_num_pool_ops(model.architecture)
    dataloaders = construct_dataloaders(
        options.dataset, batch_size, num_pool_ops, output_filenames=True
    )

    eval_cnn(model, dataloaders, options.method, options.alias, options.save_uncs)
