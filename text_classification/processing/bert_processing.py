import os
import logging
from typing import List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report

class BERTDataset(Dataset):
    def __init__(self, tokenizer, text: List[str], labels: List[str] = None):
        
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels

        logging.getLogger('transformers.tokenization_utils').setLevel(logging.FATAL)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        BERT_data = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long),
        }

        return BERT_data
    

class BERTDataBuilder:
    def __init__(
            self,
            model_name: str,
            datasets: str,
            text_column: str,
            label_column: str,
            num_labels:int=2,
            random_state: int = 1,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)        
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.num_labels = num_labels

        """
        Datasets
        """
        self.train_dataframe = datasets['train_dataframe']
        self.eval_dataframe = datasets['eval_dataframe']
        self.test_dataframe = datasets['test_dataframe']
        self.datalist = [self.train_dataframe, self.eval_dataframe, self.test_dataframe]

    def build_data(self):
        for df in self.datalist:
            df[self.label_column] = df[self.label_column].fillna(value=0)  # clear nan values from labels
            df[self.label_column] = df[self.label_column].astype(int)
            df = df[[self.label_column, self.text_column]]
            df = df[~df[self.text_column].isnull()].reset_index(drop=True)  # clear nan values from text

        print("Building Datasets...")
        print("---Training Dataset...")
        self.train_dataset = BERTDataset(tokenizer=self.tokenizer, text=self.train_dataframe[self.text_column], labels=self.train_dataframe[self.label_column])
        print("---Validation Dataset...")
        self.eval_dataset = BERTDataset(tokenizer=self.tokenizer, text=self.eval_dataframe[self.text_column], labels=self.eval_dataframe[self.label_column])
        print("---Test Dataset...")
        self.test_dataset = BERTDataset(tokenizer=self.tokenizer, text=self.test_dataframe[self.text_column], labels=self.test_dataframe[self.label_column])

class TrainingBERT:
    def __init__(self, data:BERTDataBuilder, training_args:dict):
        self.data = data
        self.model_name = self.data.model_name
        self.num_labels = self.data.num_labels
        self.training_args = TrainingArguments(**training_args)
        self.trainer = Trainer(
            model=None,
            args=self.training_args,
            train_dataset=self.data.train_dataset,
            eval_dataset=self.data.eval_dataset,
            tokenizer=self.data.tokenizer,
            model_init=self.model_init,
            compute_metrics=self.compute_metrics,
            )
        
        if not os.path.exists(training_args["output_dir"]):
            os.makedirs(training_args["output_dir"])
        
    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=self.num_labels,
        )

    def Training(self) -> None:
        print('>>> Trainer is Training...')
        self.trainer.train()

    def Testing(self) -> dict:
        prediction_data = self.generate_predictions()
        visuals = self.eval_metrics(prediction_data)
        return visuals

    def SaveModel(self, output_dir):
        print('>>> Saving Model')
        self.trainer.save_model(output_dir=output_dir)

    def generate_predictions(self) -> pd.DataFrame:
        print(">>> Generating Prediction Dataframe")
        pred_logs, label_ids, _ = self.trainer.predict(self.data.test_dataset)
        predictions = np.argmax(pred_logs, axis=-1)
        data = np.array([predictions.tolist(), label_ids.tolist()])
        columns = ['prediction', 'label_id']
        dataframe = pd.DataFrame(data=data.T, columns=columns)
        return dataframe
    
#     @staticmethod
    def compute_metrics(self, results) -> dict:
        y_true = results.label_ids
        y_pred = results.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='weighted')
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        report_dict = classification_report(y_true=y_true,y_pred=y_pred,output_dict=True)

        target_metrics = report_dict['1']
        target_precision = target_metrics['precision']
        target_recall = target_metrics['recall']
        target_f1 = target_metrics['f1-score']

        return dict(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            target_precision=target_precision,
            target_recall=target_recall,
            target_f1=target_f1,
        )
    
    @staticmethod
    def compute_objective(metrics):
        """
        Only Optimize for Given Metric
        Make sure the term 'eval_' precedes
        the metric you are trying to optimize
        """
        opt_metric = 'target_f1'
        return metrics[f'eval_{opt_metric}']

    @staticmethod
    def eval_metrics(dataframe: pd.DataFrame, label_col: str = 'label_id', pred_col: str = 'prediction') -> dict:
        print(">>> Generating Evaluation Metrics")
        y_true = dataframe[label_col]
        y_pred = dataframe[pred_col]

        confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, cmap='Blues')
        precision_recall = PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_pred)
        roc_curve = RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred)

        return dict(
            confusion_matrix=confusion_matrix,
            precision_recall=precision_recall,
            roc_curve=roc_curve
            )