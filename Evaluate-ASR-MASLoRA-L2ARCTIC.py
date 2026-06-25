import argparse
import pandas as pd
from datasets import DatasetDict, Dataset, Audio
import json
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from safetensors import safe_open
import evaluate
import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments
from MASLoRA_models import WhisperWithMASLoRAForASR
from trainer import TrainerSeq2SeqAdapterClassifASR
from Normalizer import filterAndNormalize
import jiwer

def get_args():
    parser = argparse.ArgumentParser()

    # Data related arguments
    parser.add_argument(
        "--datapath", type=str, help="Path to dataset root", required=True
    )

    parser.add_argument(
        "--metadata", type=str, help="Which metadata file to use", required=True
    )

    parser.add_argument(
        "--json", type=str, help="Which json file the accent adapters were trained with", default="label_id_map.json"
    )

    parser.add_argument(
        "--model", type=str, help="Which model to test", required=True
    )

    parser.add_argument(
        "--accent", type=str, help="Which accent to test on", required=True, choices=['arabic', 'chinese', 'hindi', 'korean', 'spanish', 'vietnamese', 'all']
    )

    parser.add_argument(
        "--top_k", type=int, help="Top_k decoding", choices=[1, 6], default=6
    )

    parser.add_argument(
        "--accent_weight_denominator", type=float, help="Accent weight denominator (max 6)", choices=[1, 1.5, 2, 3, 4, 5, 6], default=6
    )


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()


    df = pd.read_csv(f'L2-ARCTIC/metadata_{args.metadata}.csv')
    dts = 'L2-ARCTIC'
    mdl = args.model
    accent_args = args.accent


    opd = './' + mdl.split('/')[-1] + '-' + dts.split('/')[-1][:-4]
    print('Testing Model  : {}'.format(mdl))
    print('On Dataset     : {}'.format(dts))
    print('On accent      : {}'.format(accent_args))
    print('Output Dir.    : {}'.format(opd))


    path = args.datapath

    # Data loading
    df = df.rename(columns={'split': 'split', 'file_name': 'audio', 'text': 'text', 'accent': 'accent'})
    df = df[df['text'].notna()]

    test_df = df.loc[df['split'] == 'test']

    # using only the sentences and speakers never seen during training
    with open(f'L2-ARCTIC/test_speakers_sentences_{args.metadata}.json', 'r') as file: 
        test_dict = json.load(file)
        file.close()
    test_df = test_df.loc[test_df["audio"].str.startswith(tuple(test_dict["test_speakers"]))]
    test_df = test_df.loc[test_df["audio"].str.endswith(tuple(test_dict["test_sentences"]))]

    if accent_args != 'all':
        test_df = test_df.loc[test_df["accent"] == accent_args]

    for index, row in test_df.iterrows():
        file_name = row['audio']
        complete_name = path+file_name
        test_df.at[index, 'audio'] = complete_name

    print(test_df)

    dataset = DatasetDict()


    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
    dataset['test'] = test_ds

    print(dataset)

    with open(mdl+"/../args.json") as file:
        json_args = json.load(file)

   
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{json_args['whisper']}")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{json_args['whisper']}", language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{json_args['whisper']}", language="English", task="transcribe")

    with open(f"L2-ARCTIC/{json_args['json']}") as file:
        labels_to_ids = json.load(file)

    @dataclass
    class DatasetPrepper:
        feature_extractor: WhisperProcessor
        tokenizer: WhisperTokenizer
        labels_to_ids: Dict

        def __call__(self, batch):
            audio = batch["audio"]

            # compute log-Mel input features from input audio array 
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode accent label to accent id
            batch["labels"] = []
            accent = batch["accent"]
            batch["labels"].append([self.labels_to_ids[accent]])

            # encode target text to tokens
            batch["labels"].append(self.tokenizer(batch["text"], max_length=448, truncation=True).input_ids) # MAX LENGTH
            
            # adding the file_name as it can be useful for output processing
            batch["name"] = '/'.join(batch["audio"]['path'].split('/')[-3:])

            # have to use accent_class here as it is a model's kwargument
            batch["accent_class"] = []
            batch["accent_class"].append([self.labels_to_ids[accent]])

            return batch


    dataset = dataset.map(DatasetPrepper(feature_extractor, tokenizer, labels_to_ids),
                              num_proc=4, 
                              writer_batch_size=1000, 
                              remove_columns=['audio'],
                            )

    file_names = []
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"][1]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            # keeping file_names
            names = [feature["name"] for feature in features]
            for n in names: 
                file_names.append(n)
            
            # add accent labels to batch labels
            batch["labels"] = []
            batch["labels"].append(labels)
            batch["labels"].append(torch.LongTensor([feature["labels"][0][0] for feature in features]))

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load('wer')

    df = pd.DataFrame()

    def compute_metrics(pred):
        global df
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        print('pred:', pred_str)
        print('label:', label_str)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        print(f'WER = {wer}')

        df = pd.DataFrame({'hyp': pred_str, 'ref': label_str})

        return {"wer": wer}

    print(args)

    # get accent classes
    accent_classes = []
    for c in labels_to_ids:
        accent_classes.append(c)

    # create model from arguments and trained model
    model = WhisperWithMASLoRAForASR.from_pretrained(f"openai/whisper-{json_args['whisper']}",
                                                                accent_classes = accent_classes, 
                                                                lora_targets=json_args["targets"],
                                                                lora_rank=json_args["lora_rank"], 
                                                                lora_dropout=json_args["lora_dropout"], 
                                                                lora_alpha=json_args["lora_alpha"],
                                                                use_maslora_decoder=json_args["use_maslora_decoder"],
                                                                no_decoder=json_args["no_decoder"],
                                                                use_top_k = args.top_k,
                                                                accent_weight_denominator=args.accent_weight_denominator)

    # load parameters from trained model
    hf_state_dict = {}
    for k in model.state_dict():
        if "lora" not in k:
            hf_state_dict[k] = model.state_dict()[k]
    with safe_open(mdl+"/model.safetensors", framework="pt") as f:
        for k in f.keys():
            hf_state_dict[k] = f.get_tensor(k)

    model.load_state_dict(hf_state_dict, strict=True)

    model.generation_config.language = "<|en|>"
    model.generation_config.task = "transcribe"

    training_args = Seq2SeqTrainingArguments(
        output_dir='ignore',
        fp16=False,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False
    )

    trainer = TrainerSeq2SeqAdapterClassifASR(
        args=training_args,
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )


    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()

    trainer.evaluate(dataset['test'], language='en')

    df['ref-norm'] = df.apply(lambda x: filterAndNormalize(x['ref']), axis=1)
    df['hyp-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp']), axis=1)

    def calcWER(df):
        wer_cln = jiwer.wer(list(df['ref']), list(df['hyp']))
        wer_norm = jiwer.wer(list(df['ref-norm']), list(df['hyp-norm']))
        cer_cln = jiwer.cer(list(df['ref']), list(df['hyp']))
        cer_norm = jiwer.cer(list(df['ref-norm']), list(df['hyp-norm']))

        print('WER Without norm   : {} %'.format(round(wer_cln*100,4)))
        print('WER With norm      : {} %'.format(round(wer_norm*100,4)))
        print('CER Without norm   : {} %'.format(round(cer_cln*100,4)))
        print('CER With norm      : {} %'.format(round(cer_norm*100,4)))

    calcWER(df)