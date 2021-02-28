# DeepHumor: Image-based Meme Generation using Deep Learning

> Final Project in "Deep Learning" course in Skotech, 2020.  
> Authors: [Ilya Borovik](https://github.com/ilya16), [Bulat Khabibullin](https://github.com/Bulichek), [Vladislav Kniazev](https://github.com/Vladoskn), [Oluwafemi Olaleke](https://github.com/6861) and [Zakhar Pichugin](https://github.com/zakharpichugin)
>
>[![Open in YouTube](https://img.shields.io/badge/_-Presentation-red.svg?logo=youtube&labelColor=5c5c5c)](https://youtu.be/gf-HcRwsSfI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilya16/deephumor/blob/master/deephumor_demo.ipynb)

<img alt="Deep Learning meme" src="/assets/deep-learning-meme.jpg" width="480">

## Description

The repository presents multiple meme generation models (see illustrations [below](#models)):

- Captioning LSTM with Image-only Encoder
- Captioning LSTM with Image-label Encoder
- Base Captioning Transformer with Global image embedding
- Captioning Transformer LSTM with Spatial image features

**Observe the models in action in the demo notebook:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilya16/deephumor/blob/master/deephumor_demo.ipynb)
[![Open in GitHub](https://img.shields.io/badge/_-Open_in_GitHub-blue.svg?logo=Jupyter&labelColor=5c5c5c)](deephumor_demo.ipynb)

All pretrained models will be automatically downloaded and built in Colab runtime.

Except for the models, we collect and release a large-scale dataset of 900,000 meme templates crawled from [MemeGenerator](https://memegenerator.net) website.
The dataset is uploaded to [Google Drive](https://drive.google.com/file/d/1j6YG3skamxA1-mdogC1kRjugFuOkHt_A). Description of the dataset is given in the corresponding [section](#dataset).

*Note: Repository state at the end of "Deep Learning" course project is recorded in the branch* [`skoltech-dl-project`](https://github.com/ilya16/deephumor/tree/skoltech-dl-project).

## Training code

The example code for training the models is provided in [Colab notebook](https://colab.research.google.com/drive/1ayyWPuOw8ET2SRZ5KD-r4dwMH4jBn-B8). It contains the training progress and TensorBoard logs for all experiments described in the project report.

## Dataset

We crawl and preprocess a large-scale meme dataset consisting of 900,000 meme captions for 300 meme template images collected from [MemeGenerator](https://memegenerator.net) website.
During the data collection we clean the data from evident duplicates, long caption outliers, non-ASCII symbols and non-English templates.

### Download dataset
Crawled dataset of 300 meme templates with 3000 captions per templates can be downloaded
using [`load_data.sh`](load_data.sh) script or directly from [Google Drive](https://drive.google.com/file/d/1j6YG3skamxA1-mdogC1kRjugFuOkHt_A). The data is split into `train/val/test` with 2500/250/250 captions per split for each template. We provide the data splits to make the comparison of new models with our works possible.

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

## Models

### Captioning LSTM
<img alt="Captioning LSTM" src="/assets/lstm.png" width="480">

### Captioning LSTM with labels
<img alt="Captioning LSTM with labels" src="/assets/lstm-labels.png" width="480">

### Captioning Base Transformer
<img alt="Captioning Base Transformer" src="/assets/base-transformer.png" width="480">

### Captioning Transformer
<img alt="Captioning Transformer" src="/assets/transformer.png" width="480">
