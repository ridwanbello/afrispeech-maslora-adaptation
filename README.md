# MAS-LoRA

## Getting started

Code was developped and tested using Python 3.10.9.
Install the required packages using :
```
pip install -r requirements.txt
```

## Data
The dataset used in our research is **L2-ARCTIC**, which you can download for free [here](https://psi.engr.tamu.edu/l2-arctic-corpus/).
The L2-ARCTIC/ folder contains the data split for the 8-fold cross-validation.

## How to fine-tune Whisper using MAS-LoRA

Use the ```FineTuning-MASLoRA-L2ARCTIC.py``` program.
You can see in details what arguments you can use with ```--help```.
To reproduce the results obtained in our paper, use the following commands and re-run them for each metadata files corresponding to each fold :

QKVO :
```
python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qkvo --no_decoder

python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qkvo

python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qkvo --use_maslora_decoder
```
QV :
```
python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qv --no_decoder

python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qv

python FineTuning-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata metadata_1.csv --whisper small --targets qv --use_maslora_decoder
```

## How to test a fine-tuned model with MAS-LoRA

Use the ```Evaluate-ASR-MASLoRA-L2ARCTIC.py``` program.
You can see in details what arguments you can use with ```--help```.

```
python Evaluate-ASR-MASLoRA-L2ARCTIC.py --datapath path/to/dataset --metadata 1 --accent all --model path/to/model/checkpoint/ --accent_weight_denominator 6 --top_k 6
```

## Credits

This code was built on the [Transformers](https://github.com/huggingface/transformers) package and its Whisper's implementation.
The dataset used in our research is **L2-ARCTIC**, which you can download for free [here](https://psi.engr.tamu.edu/l2-arctic-corpus/).

## License

All code is licensed under the LGPL-3.0 license. See the [LICENSE](LICENSE.txt) file for details.
In this repository we use [Whisper](https://www.github.com/openai/whisper) which is licensed under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) and [Transformers](https://github.com/huggingface/transformers) which is licensed under the [Apache-2.0 License](https://github.com/huggingface/transformers/blob/main/LICENSE)