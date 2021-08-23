#Gambler's Dice

Code for the paper "[Learning to Abstain: Gambler's Dice](https://github.com/miracsan/GamblersDice)" by Mirac Sanisoglu 
and Seong Tae Kim

##Getting Started
The code supports four different datasets for experiments:
* [Covid19](https://zenodo.org/record/3757476#.Xpz8OcgzZPY)
* [MMWHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) (only the CT subset)
* [Spleen](medicaldecathlon.com)
* [SegTHOR](https://competitions.codalab.org/competitions/21145)

Current implementation expects the data to be in .npz format, where each file consists of two subfiles:
* X : The preprocessed 3D image (np.float32)
* Y : The corresponding ground-truth data (np.uint8)

Furthermore, these files are expected to lie under the directories
* *GamblersDice/data/<dataset_name>/train*
* *GamblersDice/data/<dataset_name>/val*
* *GamblersDice/data/<dataset_name>/test* 

where *dataset_name* is one of [Covid19, MM-WHS, Spleen, Thor].

The functions under *dataPreparation.py* can be used to process the raw files and put them inside the expected directories.

##How to Run
To train a network using our proposed Gambler's Dice (with an alpha hyperparameter of 0.1) on the MM-WHS dataset,
```
python3 main.py --alias 'gamblers_α_0.1_whs' --method 'gamblers_dice' --arch 'unet3d' --dataset 'whs' --alpha 0.1
``` 
The trained model and the logs are going to be saved under *GamblersDice/results/<alias>*

Similarly, to train a baseline on the same dataset using Dice loss,
```
python3 main.py --alias 'dice_whs' --method 'dice' --arch 'unet3d' --dataset 'whs' --alpha 0.1
``` 

| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| method 	| [str] 	| dice, dice_ignore, ce, gamblers_dice, cheaters_dice, fault_dice, ranking_dice | dice|
| arch 	| [str] 	| unet3d, vnet| unet3d|
| alias 	| [str] 	| folder name for the results of the current experiment| dice_covid19|
| dataset 	| [str]	| covid19, whs, spleen, thor| 	covid19|
| use-model 	| [str] 	| false, best(loads the best checkpoint under the specified alias), last(loads the last checkpoint under the specified alias)| false	|
| num-epochs	| [int] 	| train for X epochs (unless stopped early)| 300|
| lr 	| [float] 	| learning rate | 0.01  |
| weight-decay 	| [str] 	| weight decay	|  0.001 |
| optimizer 	| [int] 	| choose config 1, 2, 3, 4	|  1 |
| scheduler 	| [int] 	|  choose config 1, 2, 3 | 1	|
| alpha 	| [float] 	|  hyperparameter | 0.1	|
| lamda 	| [float] 	|  hyperparameter #2 | 0.1	|
| imp 	| [int] 	|  abort training if no improvement in the last X epochs | 15	|


##Evaluation
To obtain the Dice scores at different coverages, run *evalScript.py*, specifying the following arguments:

| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| method 	| [str] 	| softmax_response, mc_dropout, extra_class | softmax_response|
| save_uncs 	| [int] 	| whether to save the uncertainty scores as npz files for later visualization: 0 or 1 | 0|
| alias 	| [str] 	| the alias of the experiment that we want to evaluate| dice_covid19|
| dataset 	| [str]	| covid19, whs, spleen, thor| 	covid19|
| checkpoint 	| [str] 	| name of the checkpoint under the alias folder that we want to use for evaluation	|  best_checkpoint |

For example, in order to evaluate a trained model using Softmax Response and MC-Dropout, run
```
python3 evalScript.py  --method 'softmax_response' --alias dice_whs --dataset 'whs' --checkpoint 'best_checkpoint' --save-vars 1
python3 evalScript.py  --method 'mc_dropout' --alias dice_whs --dataset 'whs' --checkpoint 'best_checkpoint' --save-vars 1
``` 

The specified method will be used to get a confidence estimate for each voxel in the dataset. The coverage is then 
gradually decreased by discarding the most uncertain voxels. The results are saved under the experiment folder in csv format.

Finally, the AUC values of Dice scores can be computed using *extractAuc.py*:

| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| alias 	| [str] 	| the alias of the experiment that we want to calculate AUCs for| dice_covid19|
| covs 	| [int] 	| end-coverage values for which the AUC should be calculated	|  90 |

As an example, the following command extracts the AUC scores for the coverages [90, 100] and [95, 100]
```
python3 extractAuc.py --alias dice_covid19_4 --covs 90 95
``` 

The results are again saved under the specified experiment folder in csv format.


