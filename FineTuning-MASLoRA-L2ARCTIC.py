import argparse
import pandas as pd
from datasets import DatasetDict, Dataset, Audio
from transformers import (WhisperFeatureExtractor, 
                          WhisperTokenizer, 
                          WhisperProcessor,
                          Seq2SeqTrainingArguments,
                          )
from trainer import TrainerSeq2SeqAdapterClassifASR
from MASLoRA_models import WhisperWithMASLoRAForASR
from Normalizer import filterAndNormalize
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import json
import os

def get_args():
    parser = argparse.ArgumentParser()

    # Data related arguments
    parser.add_argument(
        "--datapath", type=str, help="Path to dataset root", required=True
    )

    parser.add_argument(
        "--metadata", type=str, help="Metadata file to use", default="metadata_1.csv"
    )

    parser.add_argument(
        "--json", type=str, help="Json file to use for the accent labels -> ids mapping", default="label_id_map.json"
    )

    parser.add_argument(
        "--whisper", type=str, help="Which whisper model to use", default="small", choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
    )

    parser.add_argument(
        "--lora_rank", type=int, help="Rank for MAS-LoRA", default=16
    )

    parser.add_argument(
        "--lora_dropout", type=float, help="Dropout for MAS-LoRA", default=0.05
    )

    parser.add_argument(
        "--lora_alpha", type=int, help="MAS-LoRA alpha (scaling factor)", default=16
    )

    parser.add_argument(
        "--use_maslora_decoder", action='store_true', help="Applies MAS-LoRA to decoder layers aswell (Default: use LoRA in decoder layers)", default=False
    )

    parser.add_argument(
        "--no_decoder", action='store_true', help="Doesn't fine-tune decoder", default=False
    )

    parser.add_argument(
        "--targets", type=str, help="MAS-LoRA targets (qv and qkvo were used in the paper)", default="qv"
    )

    parser.add_argument(
        "--batch_size", type=int, help="The size of the minibatch", default=16
    )

    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=5e-5
    )

    parser.add_argument(
        "--num_iter", type=int, help="The number of iterations to train for", default=28000
    )

    parser.add_argument(
        "--resume", help="Set to continue training a model", action='store_true', default=False
    )

    parser.add_argument(
        "--debug", help="Set output directory to ./debug/ and load a smaller dataset", action='store_true', default=False
    )

    args = parser.parse_args()

    return args

def get_output_dir_name(args):
    if args.debug:
        return './debug/'
    
    opd = './' + args.whisper + '-' + dts[:-4] + '-MASLoRA'

    if args.use_maslora_decoder:
        opd = opd+"-MASLoRADecoder"
    elif args.no_decoder:
        opd = opd+"-NoDecoder"
    else:
        opd = opd+"-ClassicLoRADecoder"

    opd = opd+f'-alpha{args.lora_alpha}'
    
    opd = opd+f'-rank{args.lora_rank}'

    t = ''
    for lt in args.targets:
        t += lt
    opd = opd + f'-{t}'

    return opd


if __name__ == "__main__":
    args = get_args()

    df = pd.read_csv(f'L2-ARCTIC/{args.metadata}')
    dts = f'L2-ARCTIC_{args.metadata}'
    path = args.datapath

    mdl = f'openai/whisper-{args.whisper}'

    opd = get_output_dir_name(args)

    # changes qkvo to q_proj, v_proj, out_proj
    lora_targets = []
    for t in args.targets:
        if t == 'o':
            t = 'out_proj'
        else:
            t = f'{t}_proj'
        lora_targets.append(t)
    args.targets = lora_targets

    if args.debug:
        opd = './debug'

    with open(f'L2-ARCTIC/{args.json}') as json_file:
        labels_to_ids = json.load(json_file)

    print('Training Model : {}'.format(mdl))
    print('On Dataset     : {}'.format(dts))
    print('Datapath       : {}'.format(path))
    print('JSON           : {}'.format(args.json))
    print('Output Dir.    : {}'.format(opd))

    batch_size = args.batch_size
    grad_acc_steps = 16 // batch_size # change here if you want to use another effective batch size
    
    # Training arguments (initialized early to use mapping in parallel)
    training_args = Seq2SeqTrainingArguments(
        output_dir=opd,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=args.lr,
        max_steps=args.num_iter,
        gradient_checkpointing=False,
        fp16=False,
        save_total_limit=1,
        evaluation_strategy="steps",
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250 if not args.debug else 10,
        eval_steps=250 if not args.debug else 10,
        logging_steps=25 if not args.debug else 2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # save program arguments to output dir
    os.makedirs(opd, exist_ok=True)
    with open(f'{opd}/args.json', 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)

    ############## DATA LOADING ##############
    for index, row in df.iterrows():
        file_name = row['file_name']
        complete_name = path+file_name
        df.at[index, 'file_name'] = complete_name

    df = df.rename(columns={'split': 'split', 'file_name': 'audio', 'text': 'text', 'accent': 'accent'})
    df = df[df['text'].notna()]

    train_df = df.loc[df['split'] == 'train'].sample(frac=1).reset_index(drop=True)
    validation_df = df.loc[df['split'] == 'validation'].sample(frac=1).reset_index(drop=True)
    
    if args.debug:
        train_df = train_df.head(10)
        validation_df = validation_df.head(10)

    dataset_train = DatasetDict()
    dataset_validation = DatasetDict()

    train_ds = Dataset.from_pandas(train_df)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000)) # Load and resample audio data to 16kHz
    dataset_train['train'] = train_ds


    validation_ds = Dataset.from_pandas(validation_df)
    validation_ds = validation_ds.cast_column("audio", Audio(sampling_rate=16000)) # Load and resample audio data to 16kHz
    dataset_validation['validation'] = validation_ds

    feature_extractor = WhisperFeatureExtractor.from_pretrained(mdl)
    tokenizer = WhisperTokenizer.from_pretrained(mdl, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(mdl, language="English", task="transcribe")

    print(dataset_train)
    print(dataset_validation)

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
            return batch

    with training_args.main_process_first(desc="Map"):
        print("Mapping dataset train")
        dataset_train = dataset_train.map(DatasetPrepper(feature_extractor, tokenizer, labels_to_ids), 
                              num_proc=1, 
                              writer_batch_size=1000, 
                              remove_columns=['audio'],
                              )


    with training_args.main_process_first(desc="Map"):
        print("Mapping dataset validation")
        dataset_validation = dataset_validation.map(DatasetPrepper(feature_extractor, tokenizer, labels_to_ids), 
                              num_proc=1, 
                              writer_batch_size=1000, 
                              remove_columns=['audio'],
                              )
        print("Done mapping dataset validation")

    dataset = DatasetDict()
    dataset["train"] = dataset_train["train"]
    dataset["validation"] = dataset_validation["validation"]

    print(dataset)

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

            # add accent labels to batch labels
            batch["labels"] = []
            batch["labels"].append(labels)
            batch["labels"].append(torch.LongTensor([feature["labels"][0][0] for feature in features]))

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    
    metric = evaluate.load('wer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [filterAndNormalize(s) for s in pred_str]
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        label_str = [filterAndNormalize(s) for s in label_str]

        print('pred:', pred_str)
        print('label:', label_str)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        print(f'WER = {wer}')
        return {"wer": wer}

    # get accent classes
    accent_classes = []
    for c in labels_to_ids:
        accent_classes.append(c)

    # create model from arguments
    model = WhisperWithMASLoRAForASR.from_pretrained(mdl, 
                                                    accent_classes=accent_classes, 
                                                    lora_targets=lora_targets, 
                                                    lora_rank=args.lora_rank, 
                                                    lora_dropout=args.lora_dropout, 
                                                    lora_alpha=args.lora_alpha,
                                                    use_maslora_decoder=args.use_maslora_decoder,
                                                    no_decoder=args.no_decoder,
                                                    accent_weight_denominator=1,
                                                    use_top_k=1
                                                    )

    model.freeze_non_lora() # freeze non LoRA parameters

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {trainable_params}')
    print(f'% of trainable params.: {trainable_params/sum(p.numel() for p in model.parameters())*100}')

    model.init_lora() # Following LoRA initialization -> https://arxiv.org/abs/2106.09685
    model.generation_config.language = "<|en|>"
    model.generation_config.task = "transcribe"

    trainer = TrainerSeq2SeqAdapterClassifASR(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    torch.cuda.empty_cache()

    import transformers
    transformers.logging.set_verbosity_info()

    trainer.train(resume_from_checkpoint = args.resume)
    trainer.save_model(opd)