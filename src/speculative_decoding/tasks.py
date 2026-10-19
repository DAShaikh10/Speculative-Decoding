"""
This module contains the datasets of the evaluated tasks.
"""

from itertools import islice

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset

from .types import Task

# The T5 task prefixes. They are what the models are fine-tuned with.
PREFIXES: dict[Task, str] = {
    Task.TRANSLATION: "translate English to German: ",
    Task.SUMMARIZATION: "summarize: ",
}


def load_inputs(task: Task, num_examples: int) -> list[str]:
    """
    Load the first `num_examples` model inputs of the test split of a task. Only inputs are needed to measure speed.

    Translation uses WMT14 de-en (newstest2014) and summarization uses CNN/DailyMail 3.0.0.
    """

    prefix: str = PREFIXES[task]

    if task == Task.TRANSLATION:
        dataset: IterableDataset = load_dataset("wmt/wmt14", "de-en", split="test", streaming=True)
        sources = (example["translation"]["en"] for example in dataset)
    else:
        dataset: IterableDataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
        sources = (example["article"] for example in dataset)

    return [prefix + source for source in islice(sources, num_examples)]
