"""
This module contains the types used in speculative decoding.
"""

import enum


class Task(enum.StrEnum):
    """
    A class that contains the tasks for which speculative decoding can be used.
    """

    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    DIALOGUE = "dialogue"  # Chat with a decoder-only model, the counterpart of the paper's LaMDA dialog task.
    UNCONDITIONAL = "unconditional"  # Text from the start token alone, the counterpart of the paper's lm1b GPT-like task.
