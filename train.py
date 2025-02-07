## -*- coding: utf-8 -*-


!pip install datasets
!pip install transformers
!pip install peft
!pip install evaluate

from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

"""# dataset"""

# sst2
# The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
#https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
dataset = load_dataset("kaggle/phishing-email-dataset")
dataset

# display % of training data with label=1
np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])

"""# model"""

model_checkpoint = 'google-bert/bert-base-multilingual-cased'

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate classification model from model_checkpoint
model = BertForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# display architecture
model

"""# preprocess data"""

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # Your prompt
    prompt = "Classify the sentiment of the following email into Spam (Positiv) and Ham (Negative): "

    # Extract text and add the prompt
    text = [prompt + text for text in examples["text"]]


    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""# evaluation"""

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

"""# Apply untrained model to text"""

# define list of examples
text_list = ["Dear client, we need you to update your billing information. Click here to proceed.", "resourceful and ingenious entertainment .", "it 's just incredibly dull .", "the movie 's biggest offense is its complete and utter lack of tension .",
             "Your recent application has been received and is under review. Thank you for applying.", "unless you are in dire need of a diesel fix , there is no real reason to see it ."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)

    print(text + " - " + id2label[predictions.tolist()])

"""# Train model"""

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['query'])

peft_config

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 16
num_epochs = 5

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

"""# Generate prediction"""

model.to('cpu')
model.push_to_hub("CyberSec_Bert_Model",private=True)
tokenizer.push_to_hub("CyberSec_Bert_Tokenizer",private=True)

text_list = ["Ihr ICloud Speicher ist fast Voll. Bitte Kliecken Sie auf den Link um zu untersuchen.","Your iCloud storage is almost full. Please click on the link", " Hallo, wie gehts dir?","Hello, how are you?"]


print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

