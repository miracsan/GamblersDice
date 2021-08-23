import os
import argparse

from trainCnn import train_cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="dice, dice_ignore, ce, gamblers_dice, cheaters_dice, fault_dice, ranking_dice",
        default="dice",
    )
    parser.add_argument(
        "--arch",
        help="unet3d, vnet",
        default="unet3d",
    )
    parser.add_argument(
        "--alias",
        help="the models will be saved under a folder with the specified name",
        default="gamblers_α_0.1_covid19",
    )
    parser.add_argument(
        "--dataset",
        help="covid19, whs, spleen, thor",
        default="covid19",
    )
    parser.add_argument("--use-model", help="false, best, last", default="false")
    parser.add_argument(
        "--num-epochs",
        help="train for X epochs (unless stopped early)",
        type=int,
        default=300,
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", help="weight decay", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", help="choose config 1, 2, 3, 4", type=int, default=1
    )
    parser.add_argument(
        "--scheduler", help="choose config 1, 2, 3", type=int, default=1
    )
    parser.add_argument("--alpha", help="hyperparameter", type=float, default=0.1)
    parser.add_argument("--lamda", help="hyperparameter", type=float, default=0.1)
    parser.add_argument(
        "--imp",
        help="abort training if no improvement in the last X epochs",
        type=int,
        default=15,
    )

    options = parser.parse_args()

    if options.use_model == "best":
        use_model = os.path.join("results", options.alias, "best_checkpoint")
    elif options.use_model == "last":
        use_model = os.path.join("results", options.alias, "checkpoint")
    elif options.use_model != "false":
        use_model = os.path.join("results", options.alias, options.use_model)
    else:
        use_model = 0

    scheduler_config = f"scheduler_config_{options.scheduler}"
    optimizer_config = f"optimizer_config_{options.optimizer}"

    model = train_cnn(
        options.method,
        options.arch,
        options.alias,
        options.dataset,
        options.num_epochs,
        options.lr,
        options.weight_decay,
        options.alpha,
        options.lamda,
        options.imp,
        optimizer_config,
        scheduler_config,
        use_model,
    )
