from typing import ClassVar, Iterable, Pattern, Mapping, Pattern
from dataclasses import dataclass, field
import re
import emoji

__author__ = "Adolfo Fernández Santamónica"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "adofersan@yahoo.es"
__status__ = "Development"


@dataclass()
class TweetPreprocessor:
    """
    Class for preprocessing tweets, normalizing its structure.

    Constructor Args:
        lang (str): Language of the tweet to be preprocessed.
            Must be supported ({getSupportedLanguages})
        user_token (str, optional): Token with which the user mentions will be
            replaced. Defaults to "[user]".
        url_token (str, optional): Token with which the urls will be replaced.
            Defaults to "[url]".
        hashtag_token (str, optional): Token with which the hashtags will be
            replaced. Defaults to None.
        emoji_token (str, optional): Token with which the emojis will be replaced.
            Defaults to [emoji].
    """

    __USER_RE: ClassVar[Pattern[str]] = re.compile(r"@[a-zA-Z0-9_]{0,15}")

    __URL_RE: ClassVar[Pattern[str]] = re.compile(
        r"((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}"
        + r"\.{1}){1,5}(?:com|co|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|"
        + r"se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)"
    )

    __HASHTAG_RE: ClassVar[Pattern[str]] = re.compile(r"\B#(\w*[a-zA-Z]+\w*)")

    __CAMEL_START: ClassVar[Pattern[str]] = re.compile(r"([A-Z]+)")
    __EMOJI_RE: ClassVar[Pattern[str]] = re.compile(r"\|([^\|]+)\|")
    __LANGS = {"es", "en"}
    __LAUGH_RE: ClassVar[Mapping[str, Mapping[str, Pattern]]] = {
        "es": {
            "regex": re.compile("[ja][ja]+aj[ja]+"),
            "replacement": "jaja",
        },
        "en": {
            "regex": re.compile("[ha][ha]+ah[ha]+"),
            "replacement": "haha",
        },
    }

    lang: str = field()
    user_token: str = field(default="[user]")
    url_token: str = field(default="[url]")
    hashtag_token: str = field(default="")
    emoji_token: str = field(default="[emoji]")

    @classmethod
    def get_supported_languages(cls) -> Iterable[str]:
        """
        Get the supported languages of the preprocessor.

        Returns:
            Iterable[str]: Supported languages of the preprocessor.
        """
        return cls.__LANGS

    def get_special_tokens(self) -> Iterable[str]:
        """
        Get the special tokens of the preprocessor (past as arguments at constructor).

        Returns:
            Iterable[str]: Special tokens of the preprocessor.
        """
        return [
            self.user_token,
            self.url_token,
            self.hashtag_token,
            self.emoji_token,
        ]

    def preprocess(
        self,
        text: str,
        normalize_hashtag: bool = True,
        normalize_emoji: bool = True,
        normalize_laugh: bool = True,
        shorten: int = 3,
    ):
        """
        Preprocess tweets, normalizing its structure.

        Args:
            text (str): Text of the tweet to be preprocessed.
            normalize_hashtag (bool, optional): If True, hashtags will be replaced
                with the hashtag_token given in the constructor, else not.
                Defaults to True.
            normalize_emoji (bool, optional): If True, emojis will be replaced
                with the emoji_token given in the constructor, else not.
                Defaults to True.
            normalize_laugh (bool, optional): If True, laughs will be replaced
                with a standard laugh, else not. For example "hahhahhaaah" will
                be replaced by "haha".
                Defaults to True.
            shorten (int, optional): Minimum number of characters of the words.
                All words with this number of characters or less will be removed.
                Defaults to 3.

        Returns:
            [str]: Text preprocessed.
        """
        text = self.__preprocess_users(text)
        text = self.__preprocess_urls(text)
        if normalize_hashtag:
            text = type(self).__HASHTAG_RE.sub(
                self.__preprocess_hashtags, text
            )
        if normalize_emoji:
            text = self.__preprocess_emoji(text)
        if normalize_laugh:
            text = self.__preprocess_laugh(text)
        text = self.__preprocess_shorten(text, shorten=shorten)
        text = text.strip()
        return text

    def __preprocess_users(self, text: str) -> str:
        """
        Replace user mentions with the user_token given in the constructor.

        Args:
            text (str): Text to be preprocessed.

        Returns:
            str: Text preprocessed.
        """
        return type(self).__USER_RE.sub(self.user_token, text)

    def __preprocess_urls(self, text: str) -> str:
        """
        Replace urls with the url_token given in the constructor.

        Args:
            text (str): Text to be preprocessed.

        Returns:
            str: Text preprocessed.
        """
        return type(self).__URL_RE.sub(self.url_token, text)

    def __preprocess_hashtags(self, text: str) -> str:
        """
        Replace hashtags with the hashtag_token given in the constructor.

        Args:
            text (str): Text to be preprocessed.

        Returns:
            str: Text preprocessed.
        """
        text = text.groups()[0]
        decamelize = lambda x: type(self).__CAMEL_START.sub(r" \1", x).strip()
        text = decamelize(text)
        text = text.lower()
        if self.hashtag_token is not None:
            text = self.hashtag_token + " " + text
        return text

    def __preprocess_emoji(self, text: str) -> str:
        """
        Replace emojis with the its meaning and surround it on both sides with
            the emoji_token given in the constructor. For example: 😂 will be
            replaced by [emoji] face with tears of joy [emoji]

        Args:
            text (str): Text to be preprocessed.

        Returns:
            str: Text preprocessed.
        """
        text = emoji.demojize(text, language=self.lang, delimiters=("|", "|"))
        emoji2text = (
            lambda emoji: f" {self.emoji_token} "
            + " ".join(emoji.groups()[0].split("_"))
            + f" {self.emoji_token} "
        )
        text = type(self).__EMOJI_RE.sub(lambda x: emoji2text(x), text)

        return text

    def __preprocess_laugh(self, text: str) -> str:
        """
        Replace laughs with a standard laugh. For example "hahhahhaaah" will
            be replaced by "haha".

        Args:
            text (str): Text to be preprocessed.

        Returns:
            str: Text preprocessed.
        """
        laugh_re = type(self).__LAUGH_RE[self.lang]["regex"]
        replacement = type(self).__LAUGH_RE[self.lang]["replacement"]

        text = laugh_re.sub(replacement, text)
        return text

    def __preprocess_shorten(self, text: str, shorten: int) -> str:
        """
        Remove all words with a number of characters less than or as indicated.

        Args:
            text (str): Text from which you want to remove the words.
            shorten (int): Minimum number of characters.
                All words with this or less characters will be removed.

        Returns:
            str: Text preprocessed without that words.
        """
        repeated_regex = re.compile(r"(.)" + r"\1" * (shorten - 1) + "+")
        text = repeated_regex.sub(r"\1" * shorten, text)
        return text
