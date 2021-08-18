import argparse

from evalCnnSegmentation import eval_cnn
from constructModel import *
from dataloaders import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="softmax_response, mc_dropout, extra_class, extra_head, entropy, random",
        default="softmax_response",
    )
    parser.add_argument("--save-vars", type=int, default=0)
    parser.add_argument(
        "--alias",
        help="under what folder name the models will be saved",
        default="baseline_segmentation",
    )
    parser.add_argument(
        "--dataset",
        help="3Dbrats2017, 1Channel3Dbrats2017, 2Dbrats2017, mosmed, covid19, cifar10, camvid",
        default="covid19",
    )
    parser.add_argument(
        "--checkpoint", help="checkpoint, best_checkpoint", default="best_checkpoint"
    )

    options = parser.parse_args()

    model_path = os.path.join("results", options.alias, options.checkpoint)
    model, _, _, _ = load_model(model_path)

    _, batch_size, _ = get_configs_from_dataset(options.dataset)
    num_pool_ops = get_num_pool_ops(model.architecture)
    dataloaders = construct_dataloaders(
        options.dataset, batch_size, num_pool_ops, output_filenames=True
    )

    eval_cnn(model, dataloaders, options.method, options.alias, options.save_vars)
