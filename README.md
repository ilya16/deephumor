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

Except for the models we collect and release a large-scale dataset of 900,000 meme templates crawled from [MemeGenerator](memegenerator.net) website.
The dataset is upload to [Google Drive](https://drive.google.com/file/d/1j6YG3skamxA1-mdogC1kRjugFuOkHt_A/view?usp=sharing). Description of the dataset is given in the corresponding [section](#dataset).

**To observe the models in action, head to the demonstraion [notebook](deephumor_demo.ipynb) or open the same notebook with the instruction directly in [Google Colab](https://colab.research.google.com/drive/1ayyWPuOw8ET2SRZ5KD-r4dwMH4jBn-B8). The latter option is better as it will automatically download all the models in Colab runtime.**

## Training code

The example code for training the models is provided in [Colab Notebook](https://colab.research.google.com/drive/1ayyWPuOw8ET2SRZ5KD-r4dwMH4jBn-B8). The notebook contains the training progress and TensorBoard logs for all experiments described in the project report.



## Dataset

The data is crawled from [memegenerator.net](memegenerator.net).

### Download dataset
Crawled dataset of 300 meme templates with 3000 captions per templates can be download
using `load_data.sh` script. The data is split into `train/val/test` 
with 2500/250/250 captions per split for each template.

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
