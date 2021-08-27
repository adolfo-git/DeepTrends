from __future__ import annotations
from typing import ClassVar, Iterable, Mapping
from dataclasses import dataclass, field
import pandas as pd
import plotly.express as px


__author__ = "Adolfo Fernández Santamónica"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "adofersan@yahoo.es"
__status__ = "Development"


@dataclass
class OutputFormatter:
    """
    Class to format the output of a TextAnalyzer.

    Constructor Args:
        text (str): Text from of which is wanted to format the exit.
        labels (Mapping[str, float]): Map with labels and their respective probabilities.
    """

    __COLORS: ClassVar[Mapping[str, list]] = {
        "sentiment": {"NEG": "red", "NEU": "gray", "POS": "green"},
        "emotion": {
            "others": "gray",
            "joy": "green",
            "sadness": "blue",
            "anger": "red",
            "surprise": "yellow",
            "disgust": "purple",
            "fear": "black",
        },
    }

    analysis: str = field()
    text: str = field()
    labels: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.analysis not in type(self).__COLORS.keys():
            raise ValueError(
                f"Field analysis must be in {type(self).__COLORS.keys()}"
            )

    def __str__(self) -> str:
        """
        Check the string representation of the analyzer output in an easy way.
            (including labels and their probabilites).

        Returns:
            str: String representation of the analysis result.
        """
        output = max(self.labels.items(), key=lambda x: x[1])[0]
        ret = f"{self.__class__.__name__}"
        formatted_labels = sorted(self.labels.items(), key=lambda x: -x[1])
        formatted_labels = [f"{k}: {v:.3f}" for k, v in formatted_labels]
        formatted_labels = "{" + ", ".join(formatted_labels) + "}"
        ret += f"(output={output}, labels={formatted_labels})"

        return ret

    def getTopClass(self) -> str:
        """
        TODO

        Returns:
            str: [description]
        """
        return max(self.labels, key=self.labels.get)

    @classmethod
    def pie(cls, out: Iterable[OutputFormatter]) -> px.pie:
        """
        TODO

        Args:
            out (Iterable[OutputFormatter]): [description]

        Returns:
            px.pie: [description]
        """
        PROB = "Frequency"
        CLASS = "Class"
        df = pd.DataFrame(
            {
                PROB: 1,
                CLASS: [x.getTopClass() for x in out],
            }
        )
        fig = px.pie(
            df,
            values=PROB,
            names=CLASS,
            color=CLASS,
            color_discrete_map=cls.__COLORS[out[0].analysis],
            title="Clasificación de los tweets"
        )
        return fig
