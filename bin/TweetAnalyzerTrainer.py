from typing import ClassVar, Mapping, Tuple, Type
from dataclasses import dataclass, field

import pandas as pd
from datasets import Dataset, Value, ClassLabel, Features

# from tqdm.auto import tqdm
from tqdm import tqdm  # for scripts

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from .TweetPreprocessor import TweetPreprocessor

__author__ = "Adolfo Fernández Santamónica"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "adofersan@yahoo.es"
__status__ = "Development"


@dataclass
class TweetAnalyzerTrainer:
    __TRAIN_PROP: ClassVar[float] = 0.8
    __TOKENIZED_MODELS: ClassVar[set] = {"vinai/bertweet-base"}
    __BATCH_SIZE: ClassVar[int] = 256
    __EVAL_BATCH_SIZE: ClassVar[int] = 16

    lang: str = field()
    id2label: Mapping[int, str] = field(default_factory=dict)
    label2id: Mapping[str, int] = field(default_factory=dict)

    xlabel: str = field(init=False)
    ylabel: str = field(init=False)
    preprocessor: Type[TweetPreprocessor] = field(init=False)
    trainer: Type[Trainer] = field(init=False)

    def __post_init__(self):
        """
        Post init method, initializes the trainer.
        """
        self.preprocessor = TweetPreprocessor(self.lang)
        self.trainer = None

    def load_dataset(
        self,
        file_path: str,
        encoding: str = "utf-8",
        xlabel: str = "tweet",
        ylabel: str = "label",
        random_state=1,
        preprocessing_args={},
    ):
        """
        Load dataset
        TODO
        Args:
            file_path (str): [description]
            encoding (str, optional): [description]. Defaults to "utf-8".
            sep (str, optional): [description]. Defaults to "\t".
            xlabel (str, optional): [description]. Defaults to "tweet".
            ylabel (str, optional): [description]. Defaults to "label".
            random_state (int, optional): [description]. Defaults to 1.
            preprocessing_args (dict, optional): [description]. Defaults to {}.

        Returns:
            [type]: [description]
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        print(f"Loading dataset: {file_path}")
        df = pd.read_csv(
            file_path,
            encoding=encoding,
        )
        df = df.sample(400000)
        df = self.__preprocess_df(df, preprocessing_args)
        self.train_df, self.test_df = self.__split_df(
            df, random_state=random_state
        )

    def __preprocess_df(
        self, df: pd.DataFrame, preprocessing_args={}
    ) -> pd.DataFrame:
        df = df.dropna(subset=[self.xlabel])
        df = df[[self.xlabel, self.ylabel]]
        for label, idx in self.label2id.items():
            df.loc[df[self.ylabel] == label, self.ylabel] = idx
        df[self.ylabel] = df[self.ylabel].astype(int)

        preprocess = lambda x: self.preprocessor.preprocess(
            x, **preprocessing_args
        )

        tqdm.pandas(desc="Preprocessing tweets")
        df.loc[:, self.xlabel] = df[self.xlabel].progress_apply(preprocess)
        return df

    def __split_df(
        self, df: pd.DataFrame, random_state: int
    ) -> Tuple[Dataset]:
        features = Features(
            {
                self.xlabel: Value("string"),
                self.ylabel: ClassLabel(
                    num_classes=len(self.id2label),
                    names=[
                        self.id2label[k] for k in sorted(self.id2label.keys())
                    ],
                ),
            }
        )

        print("Splitting dataset...")
        train_df, test_df = train_test_split(
            df,
            stratify=df[self.ylabel],
            random_state=random_state,
            train_size=type(self).__TRAIN_PROP,
        )

        train_df = Dataset.from_pandas(train_df, features=features)
        print(f"Train dataset: {train_df.num_rows} obs.")

        test_df = Dataset.from_pandas(test_df, features=features)
        print(f"Test dataset: {test_df.num_rows} obs.")
        return train_df, test_df

    def load_model(self, base_model, max_length=128):
        """
        Loads model and tokenizer
        TODO
        """
        print(f"Loading model: {base_model}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, return_dict=True, num_labels=len(self.id2label)
        )
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.model_max_length = max_length

        if base_model not in type(self).__TOKENIZED_MODELS:
            tokenizer.add_tokens(self.preprocessor.get_special_tokens())
            self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        print(self.train_df.column_names)
        self.train_df = self.train_df.map(
            self.__tokenize,
            batched=True,
            batch_size=type(self).__BATCH_SIZE,
        )
        self.train_df = self.__format_df(self.train_df)

        self.test_df = self.test_df.map(
            self.__tokenize,
            batched=True,
            batch_size=type(self).__EVAL_BATCH_SIZE,
        )
        self.test_df = self.__format_df(self.test_df)

    def __format_df(self, df):
        df = df.map(lambda examples: {"labels": examples[self.ylabel]})
        df.set_format(
            type="torch",
            columns=[
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "labels",
            ],
        )
        return df

    def __tokenize(self, batch):
        """

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.tokenizer(
            batch[self.xlabel], padding="max_length", truncation=True
        )

    def train(self, epochs: int = 4):
        """
        TODO

        Args:
            epochs (int, optional): [description]. Defaults to 10.
        """
        total_steps = (epochs * len(self.train_df)) // (
            type(self).__BATCH_SIZE
        )
        warmup_steps = total_steps // 10
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=type(self).__BATCH_SIZE,
            per_device_eval_batch_size=type(self).__EVAL_BATCH_SIZE,
            warmup_steps=warmup_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_eval=False,
            weight_decay=0.01,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.__compute_metrics,
            train_dataset=self.train_df,
            eval_dataset=self.test_df,
        )
        print(self.train_df.column_names, flush=True)
        print(self.test_df.column_names, flush=True)
        self.trainer.train()

    def __compute_metrics(self, pred):
        """
        TODO

        Args:
            pred ([type]): [description]

        Returns:
            [type]: [description]
        """
        ret = {}
        f1s = []
        precs = []
        recalls = []
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        for i, cat in enumerate(self.label2id):
            cat_labels, cat_preds = labels == i, preds == i
            precision, recall, f1, _ = precision_recall_fscore_support(
                cat_labels, cat_preds, average="binary"
            )
            f1s.append(f1)
            precs.append(precision)
            recalls.append(recall)
            ret[cat.lower() + "_f1"] = f1

        ret["macro_f1"] = torch.Tensor(f1s).mean()
        ret["macro_precision"] = torch.Tensor(precs).mean()
        ret["macro_recall"] = torch.Tensor(recalls).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        ret["f1"] = f1
        ret["acc"] = accuracy_score(labels, preds)

        return ret

    def evaluate(self):
        """
        TODO
        """
        self.trainer.evaluate(self.test_df)

    def save_model(self, path: str):
        """
        TODO

        Args:
            path (str): [description]
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
