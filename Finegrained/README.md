# Fine-grained Classification


## Usage
### Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

### Train

To train TransFG on Serifs dataset, please refer to the `run_train.sh` script and update the `--data_root` and `--pretrained_dir` arguments. 

