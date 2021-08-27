import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass, field
from datasets import Dataset
from typing import ClassVar, Iterable, Mapping, Any, Union
from .OutputFormatter import OutputFormatter
from .TweetPreprocessor import TweetPreprocessor

__author__ = "Adolfo Fernández Santamónica"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "adofersan@yahoo.es"
__status__ = "Development"


@dataclass()
class TweetAnalyzer:
    """
    Class which can analyze subjective aspects of a tweet like the expected
    emotion or sentiments that users could perceive.

    Constructor Args:
        lang (str):
        user_token (str, optional): Token with which the user mentions will be
            replaced. Defaults to "[user]".
        url_token (str, optional): Token with which the urls will be replaced.
            Defaults to "[url]".
    """

    __MODELS: ClassVar[Mapping[str, Mapping[str, Mapping[str, str]]]] = {
        "sentiment": {
            "es": {
                "model_name": "finiteautomata/beto-sentiment-analysis",
            },
            "en": {
                "model_name": "finiteautomata/bertweet-base-sentiment-analysis",
                # BerTweet uses different preprocessing args
                "kwargs": {"user_token": "@USER", "url_token": "HTTPURL"},
            },
        },
        "emotion": {
            "es": {
                "model_name": "finiteautomata/beto-emotion-analysis",
            },
            "en": {
                "model_name": "finiteautomata/bertweet-base-emotion-analysis",
                "kwargs": {"user_token": "@USER", "url_token": "HTTPURL"},
            },
        },
    }

    analysis: str = field()
    lang: str = field()
    model_name: str = field(default=None)
    batch_size: int = field(default=32)
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def get_supported_analysis(cls) -> Iterable[str]:
        """
        Check the types of analysis supported by the analyzer.

        Returns:
            Iterable[str]: List of supported analyzes.
        """
        return TweetAnalyzer.__MODELS.keys()

    @classmethod
    def get_supported_languages(cls, analysis: str) -> Iterable[str]:
        """
        Check The Languages Supported by The Analyzer.

        Args:
            analysis (str): type of analysis you want to make.

        Returns:
            Iterable [str]: List of languages supported by said analysis.
        """
        return TweetAnalyzer.__MODELS[analysis].keys()

    @classmethod
    def __get_model_info(cls, analysis: str, lang: str) -> str:
        """[summary]

        Args:
            analysis (str): [description]
            lang (str): [description]

        Returns:
            str: [description]
        """
        a = TweetAnalyzer.get_supported_analysis()
        if analysis not in a:
            raise ValueError(f"{analysis} must be in {a}")

        l = TweetAnalyzer.get_supported_languages(analysis)
        if lang not in l:
            raise ValueError(f"{lang} must be in {l}")
        return TweetAnalyzer.__MODELS[analysis][lang]

    def __post_init__(self):
        """
        Post init method, initializes the classification model.
        """
        if not self.model_name:
            model_info = TweetAnalyzer.__get_model_info(
                self.analysis, self.lang
            )
            self.model_name = model_info["model_name"]
            self.kwargs = model_info.get("kwargs", {})
        self.__preprocessor = TweetPreprocessor(self.lang)
        self.__tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__tokenizer.model_max_length = 128
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.id2label = self.model.config.id2label

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=".",
                per_device_eval_batch_size=self.batch_size,
            ),
            data_collator=DataCollatorWithPadding(
                self.__tokenizer, padding="longest"
            ),
        )
        self.output = OutputFormatter

    def predict(
        self, tweet: Union[str, Iterable[str]]
    ) -> Union[str, Iterable[str]]:
        """
        Predicts the probabilities of inclusion to each of the groups of the
        classification indicated in the constructor for past tweets as an argument.

        Args:
            tweet (Union[str, Iterable[str]]): Single or batch of tweets to be predicted.

        Returns:
            [str]: Labels with the probabilities.
        """
        if isinstance(tweet, str):
            return self.__predict_single(tweet)
        sentences = [
            self.__preprocessor.preprocess(sent, **self.kwargs)
            for sent in tweet
        ]
        dataset = Dataset.from_dict({"text": sentences})
        dataset = dataset.map(
            self.__tokenize, batched=True, batch_size=self.batch_size
        )
        output = self.trainer.predict(dataset)
        logits = torch.tensor(output.predictions)

        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = [
            {self.id2label[i]: probs[j][i].item() for i in self.id2label}
            for j in range(probs.shape[0])
        ]
        probs = [
            self.output(self.analysis, sent, prob)
            for sent, prob in zip(sentences, probs)
        ]
        return probs

    def __predict_single(self, tweet: str) -> str:
        """
        Predicts the probabilities of inclusion to each of the groups of the
        classification indicated in the constructor for single past tweet as an argument.

        Args:
            tweet (str): Tweet to be predicted.

        Returns:
            [str]: Labels with the probabilities.
        """
        device = self.trainer.args.device
        sentence = self.__preprocessor.preprocess(tweet, **self.kwargs)
        idx = (
            torch.LongTensor(
                self.__tokenizer.encode(
                    sentence,
                    truncation=True,
                    max_length=self.__tokenizer.model_max_length,
                )
            )
            .view(1, -1)
            .to(device)
        )
        output = self.model(idx)
        probs = torch.nn.functional.softmax(output.logits, dim=1).view(-1)
        probs = {self.id2label[i]: probs[i].item() for i in self.id2label}
        probs = self.output(self.analysis, sentence, probs)
        return probs

    def __tokenize(self, batch: Mapping[str, Iterable[str]]):
        return self.__tokenizer(batch["text"], padding=False, truncation=True)
