# DeepHumor
Meme generation using Deep Learning

## Installation
```shell script
git clone https://github.com/ilya16/deephumor
cd deephumor
pip install -r requirements.txt
```

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

Then, split the data using:
```shell script
python split_data.py --data-dir ../memes
```


