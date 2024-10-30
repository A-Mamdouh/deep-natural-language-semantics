"""This module contains the implmenetaion of the WordEncoder class
used to transform string text to a numerical tensor"""

from transformers import BertTokenizer, BertModel
import torch


class WordEncoder:  # pylint: disable=R0903:too-few-public-methods
    """The word encoder produces tensor encodings from text.
    This is used for the learned heuristic agent"""

    word_encoding_length = 768

    def __init__(self, device: str = "cuda"):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(device)
        self.device: str = device

    @property
    def output_size(self) -> int:
        return self.model.config.hidden_size

    def encode_word(self, word: str) -> torch.Tensor:
        """Return a tensor encoding of the input word"""
        tokens = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
        tokens = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokens.items()
        }
        with torch.no_grad():
            raw_outputs = self.model(**tokens)
            encoding = raw_outputs.last_hidden_state.mean(dim=1).squeeze()
        return encoding


if __name__ == "__main__":
    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load pre-trained model
    model = BertModel.from_pretrained("bert-base-uncased")

    # Encode text
    text: str = "walk"
    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(encoded_input)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()

    # Example usage
    print(last_hidden_states.shape)
