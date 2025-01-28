from openpyxl.styles.builtins import output
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,)

import torch

class EmailClassifierLora:
    def __init__(self, email, betreff, url, absender):
        self.email=email
        self.betreff=betreff
        self.url=url
        self.absender=absender
        self.tokenizer = AutoTokenizer.from_pretrained("RandomAB/CyberSec_Bert_Tokenizer3")
        self.model_checkpoint = 'RandomAB/CyberSec_Bert_Model3'

        # define label maps
        self.id2label = {0: "Ham", 1: "Spam"}
        self.label2id = {"Ham": 0, "Spam": 1}
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=2, id2label=self.id2label, label2id=self.label2id)
        self.model.to('cpu')
        # add pad token if none exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def get_attributes_as_list(self):
        return [self.email, self.betreff, self.absender, self.url]




    # generate classification model from model_checkpoint


    # define list of examples
    text_list = ["Dear client, we need you to update your billing information. Click here to .", "resourceful and ingenious entertainment .", "it 's just incredibly dull .", "the movie 's biggest offense is its complete and utter lack of tension .",
                 "Click on this link.This is Spam", "unless you are in dire need of a diesel fix , there is no real reason to see it ."]

    text_list2 = ["Das ist eine Spam Nachricht","Your iCloud storage is almost full. Please click on the link", " Hallo, wie gehts dir?","Hello, how are you?"]


    print("Untrained model predictions:")
    print("----------------------------")
    @staticmethod
    def classify(self, email, betreff, url, absender):

        ausgabe=list()
        for text in self.get_attributes_as_list():
            inputs = self.tokenizer.encode(text, return_tensors="pt").to("cpu")
            logits = self.model(inputs).logits
            prediction = torch.argmax(logits)
            ausgabe.append(text + " - " + self.id2label[prediction.tolist()])
            print(text + " - " + self.id2label[prediction.tolist()])

        return ausgabe