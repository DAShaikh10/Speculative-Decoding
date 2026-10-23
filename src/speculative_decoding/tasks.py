"""
This module contains the datasets of the evaluated tasks.
"""

from itertools import islice

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset

from .types import Task

# A model input: the text of a T5 task, or the chat messages of a dialogue, which the chat template turns into text.
ModelInput = str | list[dict[str, str]]

# The longest dialogue prompt in characters, which is about 250 tokens.
MAX_PROMPT_CHARS: int = 1000

# The T5 task prefixes. They are what the models are fine-tuned with.
PREFIXES: dict[Task, str] = {
    Task.TRANSLATION: "translate English to German: ",
    Task.SUMMARIZATION: "summarize: ",
}


def load_inputs(task: Task, num_examples: int | None = None) -> list[ModelInput]:
    """
    Load the first `num_examples` model inputs of the test split of a task, or all of them if it is None. Only inputs
    are needed to measure speed.

    Translation uses WMT14 de-en (newstest2014), summarization uses CNN/DailyMail 3.0.0, and dialogue uses the first
    user message of the conversations of the UltraChat test split. The paper's dialog prompts are not public.
    """

    if task == Task.DIALOGUE:
        chats: IterableDataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
        # Prompts that fit the input limit, since cutting a long one would remove the question it asks.
        first_turns = (
            [example["messages"][0]]
            for example in chats
            if example["messages"][0]["role"] == "user" and len(example["messages"][0]["content"]) <= MAX_PROMPT_CHARS
        )
        return list(islice(first_turns, num_examples))

    prefix: str = PREFIXES[task]

    if task == Task.TRANSLATION:
        dataset: IterableDataset = load_dataset("wmt/wmt14", "de-en", split="test", streaming=True)
        sources = (example["translation"]["en"] for example in dataset)
    else:
        dataset: IterableDataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
        sources = (example["article"] for example in dataset)

    return [prefix + source for source in islice(sources, num_examples)]
