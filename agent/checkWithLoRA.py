from fastapi.logger import logger
from openpyxl.styles.builtins import output
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,)

import torch


class EmailClassifierLora:
    def __init__(self):#, email, betreff, url, absender):
        #self.email=email
        #self.betreff=betreff
        #self.url=url
        #self.absender=absender
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
        return ["email", "betreff", "absender", "url"]




    # generate classification model from model_checkpoint


    # define list of examples
    text_list = ["Dear client, we need you to update your billing information. Click here to .", "resourceful and ingenious entertainment .", "it 's just incredibly dull .", "the movie 's biggest offense is its complete and utter lack of tension .",
                 "Click on this link.This is Spam", "unless you are in dire need of a diesel fix , there is no real reason to see it ."]

    text_list2 = ["Das ist eine Spam Nachricht","Your iCloud storage is almost full. Please click on the link", " Hallo, wie gehts dir?","Hello, how are you?"]


    print("Untrained model predictions:")
    print("----------------------------")
    @staticmethod
    def classify(self, email, betreff, url, absender):
        ausgabe=[]
        inhalt_inputs=self.tokenizer(email, return_tensors="pt", truncation=True, padding=True)
        betreff_inputs = self.tokenizer(betreff, return_tensors="pt", truncation=True, padding=True)
        absender_inputs = self.tokenizer(absender, return_tensors="pt", truncation=True, padding=True)
        url_inputs = self.tokenizer(url, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            inhalt_prob=torch.nn.functional.softmax(self.model(**inhalt_inputs).logits,dim=-1)[:,1].item()
            betreff_prob = torch.nn.functional.softmax(self.model(**betreff_inputs).logits, dim=-1)[:, 1].item()
            absender_prob = torch.nn.functional.softmax(self.model(**absender_inputs).logits, dim=-1)[:, 1].item()
            url_prob = torch.nn.functional.softmax(self.model(**url_inputs).logits, dim=-1)[:, 1].item()

        final_score= (0.2 * betreff_prob) + (0.4 * inhalt_prob) + (0.2 * absender_prob) + (0.2 * url_prob)
        classification="Phishing" if final_score >= 0.5 else "No Phishing"
        ausgabe.append({
            "subject": betreff,
            "body": email,
            "sender": absender,
            "subject_prob": betreff_prob,
            "body_prob": inhalt_prob,
            "sender_prob": absender_prob,
            "final_score": final_score,
            "classification": classification
        })


        return ausgabe