from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.final_environment.logic import Axioms, Predicates
from src.final_environment.natural_language import (
    Sentence,
    tableau_to_sentence,
    tableau_to_sentence_str,
)
from src.logic.base.calculus import generate_models
from src.logic.base.syntax import Constant
from src.logic.base.tableau import Tableau


@dataclass
class ModelTreeNode:
    reading_model: Tableau
    previous_sentence_reading: Optional["DialogTreeNode"] = None
    next_sentence_readings: List["DialogTreeNode"] = field(default_factory=list)

    def extend_tree(self, *sentences: Sentence) -> None:
        if not sentences:
            return
        next_sentence, *rest_sentences = sentences
        for reading in next_sentence.get_tableaus():
            node = DialogTreeNode(next_sentence, reading, parent_reading_model=self)
            node.extend_tree(*rest_sentences)
            self.next_sentence_readings.append(node)

    def get_human_str(self, rename_variables: bool = False) -> str:
        series: List[ModelTreeNode] = [self]
        current_model = self

        while current_model.previous_sentence_reading:
            current_model = current_model.previous_sentence_reading.parent_reading_model
            series.append(current_model)

        collapsed_tableaus = []
        for i, model in enumerate(series[:-1]):
            parent = series[i + 1]
            collapsed_tableaus.append(
                model.reading_model.collapse(until=parent.reading_model)
            )
        sentences = [
            tableau_to_sentence_str(tableau, rename_variables=rename_variables)
            for tableau in collapsed_tableaus
        ]
        sentences = [f for f in sentences if f]
        model_str = ". ".join(reversed(sentences))
        return f"{model_str}."

    def to_sentences(self) -> List[Sentence]:
        series: List[ModelTreeNode] = [self]
        current_model = self

        while current_model.previous_sentence_reading:
            current_model = current_model.previous_sentence_reading.parent_reading_model
            series.append(current_model)

        collapsed_tableaus = []
        for i, model in enumerate(series[:-1]):
            parent = series[i + 1]
            collapsed_tableaus.append(
                model.reading_model.collapse(until=parent.reading_model)
            )
        return [
            tableau_to_sentence(tableau) for tableau in reversed(collapsed_tableaus)
        ]

    @classmethod
    def create_tree(cls, *sentences: Sentence) -> "ModelTreeNode":
        node = ModelTreeNode(Tableau())
        node.extend_tree(*sentences)
        return node


@dataclass
class DialogTreeNode:
    sentence: Sentence
    sentence_reading: Tableau
    parent_reading_model: ModelTreeNode
    models: List[ModelTreeNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.models:
            self._gen_models()

    def _gen_models(self) -> None:
        extended_tableau = Tableau.merge(
            self.sentence_reading, parent=self.parent_reading_model.reading_model
        )
        for model in generate_models(extended_tableau, axioms=Axioms.get_axioms()):
            self.models.append(ModelTreeNode(model, self))

    def extend_tree(self, *sentences: Sentence) -> None:
        for model in self.models:
            model.extend_tree(*sentences)


@dataclass
class Dialog:
    sentences: List[Sentence]
    model_root: ModelTreeNode = field(init=False)

    def __len__(self) -> int:
        return len(self.sentences)

    def __item__(self, depth: int) -> None:
        return self.get_models(depth)

    def __post_init__(self) -> None:
        self.model_root = ModelTreeNode.create_tree(*self.sentences)

    def get_models(self, sentence_depth: Optional[int] = None) -> List[ModelTreeNode]:
        """Get all the models of the dialog at given sentence depth"""
        if sentence_depth is None:
            sentence_depth = len(self) - 1
        if sentence_depth >= len(self):
            raise KeyError(
                f"Cannot get model at depth {sentence_depth} from a dialog of depth {len(self)}"
            )
        current_models = [self.model_root]
        for _ in range(sentence_depth + 1):
            next_models = []
            for model in current_models:
                for reading in model.next_sentence_readings:
                    next_models.extend(reading.models)
            current_models = next_models
        return current_models

    @classmethod
    def from_dict(cls, raw_dialog: List[Dict[str, Any]]) -> "Dialog":
        return cls(sentences=list(map(Sentence.from_dict, raw_dialog)))


def get_unique_models(models: List[ModelTreeNode]) -> List[ModelTreeNode]:
    seen = set()
    output = []
    for model in models:
        mstr = model.get_human_str(True)
        if mstr in seen:
            continue
        seen.add(mstr)
        output.append(model)
    return output
