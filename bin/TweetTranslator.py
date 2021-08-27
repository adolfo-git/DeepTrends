#!pip install googletrans-4.0.0rc1
from googletrans import Translator
from tqdm.auto import tqdm
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
import numpy as np
import sys

__author__ = "Adolfo Fernández Santamónica"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "adofersan@yahoo.es"
__status__ = "Development"


class TweetTranslator:
    """
    Class for translating tweets. This class uses googletrans module.
    It is recommended to install googletrans-4.0.0rc1 version.
    """

    __URLS = [
        "translate.google.com",
        "translate.google.co.kr",
    ]
    __BAR_DESC = "Translation progress"
    __BAR_FORMAT = "{l_bar}{bar} [time left: {remaining}]"

    def __init__(self, tweets: pd.Series) -> None:
        """
        Initializes the tweet translator with the indicated tweet series.

        Args:
            tweets (pd.Series): Tweet Series to be translated.
        """
        self.__tweets = tweets
        self.__trans_tweets = pd.Series(
            np.nan, index=[i for i in range(0, len(tweets))]
        )
        self.__translator = Translator(service_urls=TweetTranslator.__URLS)

    def translate(self, file_path: str, src: str, dest: str, batch_size: int = 25):
        """
        Translate the Tweets of the source language to the destination language.
        See ISO 639-1. The Google API gives many problems if many packages are sent,
        the best solution so far is to avoid tweet packets that do not translate.

        Args:
            file_path (str): Route of the file where you want to store the translation.
                (Recommended to use csv format).
            src (str): Source language.
            dest (str): Destination language.
            batch (int): Batch size of the tweets that will be translated
            each iteration. Not recommended to be more than 30 because Google API
            probably throws a request exception.
        """

        i = 0
        ini = 0
        end = len(self.__tweets)
        with tqdm(
            total=end / batch_size,
            desc=TweetTranslator.__BAR_DESC,
            bar_format=TweetTranslator.__BAR_FORMAT,
        ) as bar:
            for i in range(ini, end, batch_size):
                k = min(i + batch_size, end)
                batch = range(i, k)
                tweets = self.__tweets.iloc[batch].to_string(
                    header=False, index=False
                )
                try:
                    trans = self.__translator.translate(
                        tweets, src=src, dest=dest
                    )
                    self.__trans_tweets.update(
                        pd.Series(
                            trans.text.split("\n"),
                            index=batch,
                        )
                    )
                except Exception:
                    print(sys.exc_info()[0], flush=True)
                    continue
                finally:
                    bar.update(1)

        df = self.__trans_tweets
        df.to_csv(file_path, index=False, header=["tweet"])
