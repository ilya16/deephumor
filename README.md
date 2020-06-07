# DeepHumor: Image-based Meme Generation using Deep Learning

> Final Project in "Deep Learning" course in Skotech, 2020.  
> Authors: [Ilya Borovik](https://github.com/ilya16), [Bulat Khabibullin](https://github.com/MrWag2), [Vladislav Kniazev](https://github.com/Vladoskn), [Oluwafemi Olaleke](https://github.com/6861) and [Zakhar Pichugin](https://github.com/zakharpichugin)
>
> [Video with the presentation](https://youtu.be/gf-HcRwsSfI)  

## Description

The repository contains the code for multiple meme generation models:

- Captioning LSTM with Image-only Encoder
- Captioning LSTM with Image-label Encoder
- Base Captioning Transformer with Global image embedding
- Captioning Transformer LSTM with Spatial image features

Except for the models, we collect and release a large-scale dataset of 900,000 meme templates crawled from [MemeGenerator](https://memegenerator.net) website.
The dataset is upload to [Google Drive](https://drive.google.com/file/d/1j6YG3skamxA1-mdogC1kRjugFuOkHt_A). Description of the dataset is given in the corresponding [section](#dataset).

**To observe the models in action, head to the demonstraion [notebook](deephumor_demo.ipynb) or open the same notebook with the instruction directly in [Google Colab](https://colab.research.google.com/drive/12KxXF_ch-DapDklf_AxHGU-X9RfXLvtR). The latter option is better as it will automatically download all the models in Colab runtime.**

## Training code

The example code for training the models is provided in [Colab Notebook](https://colab.research.google.com/drive/1ayyWPuOw8ET2SRZ5KD-r4dwMH4jBn-B8). The notebook contains the training progress and TensorBoard logs for all experiments described in the project report.

## Dataset

We crawl and preprocess a large-scale meme dataset consisting of 900,000 meme captions for 300 meme template images collected from [MemeGenerator](https://memegenerator.net) website.
During the data collection we clean the data from evident duplicates, long caption outliers, non-ASCII symbols and non-English templates.

### Download dataset
Crawled dataset of 300 meme templates with 3000 captions per templates can be download
using `load_data.sh` script or directly from [Google Drive](https://drive.google.com/file/d/1j6YG3skamxA1-mdogC1kRjugFuOkHt_A). The data is split into `train/val/test` with 2500/250/250 captions per split for each template. We provide the data splits to make the comparison of new models with our works possible.

The dataset archive follows the following format:

```
├── memes900k
|   ├── images -- template images
|       ├── cool-dog.jpg
|       ├── dogeee.jpg
|       ├── ...
|   ├── tempaltes.txt -- template labels and image urls
|   ├── captions.txt -- all captions
|   ├── captions_train.txt -- training split
|   ├── captions_val.txt -- validation split
|   ├── captions_test.txt -- test split
```

### Crawl dataset
To crawl own dataset, run the following script:
```shell script
python crawl_data.py --source memegenerator.net --save-dir ../memes \
    --poolsize 25 --num-templates 300 --num-captions 3000 \
    --detect-english --detect-duplicates \
    --min-len 10 --max-len 96 --max-tokens 31
```

Then, split the data into `train/val/test` using:
```shell script
python split_data.py --data-dir ../memes --splits 2500 250 250
```
