# Eye In The Sky

This is a submission for Eye In The Sky Competetion as a part of Inter-IIT tech meet hosted by IIT Bombay. 
## Prerequisites

```json
torch
tensorboardX
tensorflow
scikit-image
scikit-learn
tifffile
skimage
numpy 
argparse
gflags

```
## Installing
Basic dependencies stated in requirements.txt

To install dependencies: `pip install -r requirements.txt `

## File Description
1.aug.py - applies augmentation to the training data and saves it in a folder named image
```bash
python aug.py --path=[train folder] --dest=[dest folder]
```
2. Patches.py
creates patches of the given shape and saves them to a specified location
```bash
python patches.py --source=[source folder] --dest=[destiantion folder] --height=256 --width=256 --stride=0.5 --mode=train
```
3. train.py - trains model
```bash
python train.py --id=[experiment id] --epochs=25 --resume=[restart training(true/false)] --batch_size=16 --lr=0.001 --tag=[tag for tensorboard runs] --gpu=[true/false]
```


## How to run with default parameters


### A typical top-level directory layout

    .
    ├── train                   # Contains original sat and gt files 
    ├── image                    # contains images after augmentation
    ├── data                     # contains patches 
    .
    .         
    └── README.md
### copy train folder as given and should have sat/ and gt/ folder 
#### Then run the following process
process:
```bash
python aug.py
python channel9.py
python patches.py
python train.py --id=[] --tag=[]
python predict.py --id=[] --sub_id=[]
python stitch.py --id=[] --su_id=[]
```

### For repeating the predictions(Final Model)
1. Copy test folder as given
```bash
2. Run python patches.py --mode='test'
3. Then python predict.py --id=1 --sub_id=1
4. Then python stitch.py --id=1 --sub_id=1
```
Final Predictions will be saved in TestStichedid_1


### Instrucions for binary Models (Not Our Final Model)
Note - Copy given dataset folder as The-Eye-in-the-sky-dataset, The-Eye-in-the-sky-test-dataset in the Binary folder
```bash
1. python patches.py
2. python create_binary_mask.py
3. python train.py --id=[] 
4. python predict.py --id=[] --sub_id=[]
5. python stich_mask.py --id=[] --su_id=[]
```