import bisect
from collections import defaultdict
import json
from typing import DefaultDict, List, Optional, Set
import rich

import ollama
import rich.pretty

from src.final_environment.model_generation import ModelTreeNode, get_unique_models
from src.final_environment.model_generation import Dialog


def order_helper(dialog_str: str, models: List[ModelTreeNode]) -> List[int]:
    models_str = ""
    for i, model in enumerate(models):
        models_str += f"\n  {i+1}- {model.get_human_str(True)}"
    prompt = f"""Given the short story: "{dialog_str}", order the interpretations using their numbers using common logic and preferring known names than individuals: {models_str}"""
    # prompt = f"""Given the short story: "{dialog_str}",  Order the following interpretations from most to least reasonable. Defer resolving anaphora with random individuals: {models_str}"""
    sys_prompt = (
        """Answer using the schema { "order": [Number], "reasoning": str } """
        f"""where order is a list of {len(models)} indices answering the prompt and the reasoning is a brief string explaining your answer"""
    )
    # print(prompt)
    params = {
        "system": sys_prompt,
        "prompt": prompt,
        "model": "llama3.1",
        "format": "json",
        "options": {"temperature": 0, "seed": 42},
    }
    response = ollama.generate(
        **params,
    )
    s = json.loads(response["response"])
    # rich.pretty.pprint(s)
    missing = set(range(len(models)))
    output_order = []
    for i in s["order"]:
        index = i - 1
        if index not in missing:
            continue
        missing.discard(index)
        if index < 0 or index >= len(models):
            continue
        output_order.append(index)
    output_order.extend(missing)
    return output_order


class OrderForest:
    def __init__(self):
        self.forest: DefaultDict["ComparableModel", Set["ComparableModel"]] = (
            defaultdict()
        )

    def is_smaller(self, first, second) -> Optional[bool]:
        if self._find(first, second):
            self.set_less_than(first, second, True)
            return True
        if self._find(second, first):
            self.set_less_than(first, second, False)
            return False
        return None

    def _find(self, key: "ComparableModel", value: "ComparableModel") -> bool:
        current_level: List[ComparableModel] = self.forest.get(key, set())
        visited = set()
        while current_level:
            next_level = []
            for node in current_level:
                visited.add(node)
                if node == value:
                    return True
                next_level.extend(self.forest.get(node, set()))
            current_level = next_level
            current_level = [s for s in current_level if s not in visited]
        return False

    def set_less_than(
        self, first: "ComparableModel", second: "ComparableModel", is_less_than: bool
    ) -> None:
        if is_less_than:
            record = self.forest.get(first, set())
            record.add(second)
            self.forest[first] = record
        else:
            record = self.forest.get(second, set())
            record.add(first)
            self.forest[second] = record


class ComparableModel:
    _order_forest: OrderForest = OrderForest()

    def __init__(self, dialog_str: str, model: ModelTreeNode):
        self.dialog_str = dialog_str
        self.model = model

    def __eq__(self, other):
        if isinstance(other, ComparableModel):
            return (
                self.dialog_str == other.dialog_str
                and self.model.get_human_str() == other.model.get_human_str()
            )
        return False

    @staticmethod
    def check_cache(
        first: "ComparableModel", second: "ComparableModel"
    ) -> Optional[bool]:
        return ComparableModel._order_forest.is_smaller(first, second)

    def __lt__(self, other) -> bool:
        if not isinstance(other, ComparableModel):
            raise TypeError()
        if self.dialog_str != other.dialog_str:
            raise ValueError()
        if self == other:
            return False
        answer = ComparableModel.check_cache(self, other)
        if answer is None:
            answer = order_helper(self.dialog_str, [other.model, self.model])[0] == 0
            ComparableModel._order_forest.set_less_than(self, other, answer)
        return answer

    def __hash__(self) -> int:
        return self.model.reading_model.__hash__()


def normal_order(
    dialog_str: Dialog, models: List[ModelTreeNode]
) -> List[ModelTreeNode]:
    output_order = order_helper(dialog_str, models)
    return [models[i] for i in output_order]


def pair_order(dialog_str: Dialog, models: List[ModelTreeNode]) -> List[ModelTreeNode]:
    """Return an ordered list of reading models according to the heuristic"""
    first_model, *comparable_models = [
        ComparableModel(dialog_str, model) for model in models
    ]
    sorted_list = [first_model]
    for model in comparable_models:
        bisect.insort(sorted_list, model)
    return [m.model for m in reversed(sorted_list)]


def order_models(dialog: Dialog) -> List[ModelTreeNode]:
    """Return an ordered list of reading models according to the heuristic"""
    dialog_str = ". ".join(s.sentence for s in dialog.sentences) + "."
    models = list(dialog.get_models())
    models = get_unique_models(models)
    models = list(sorted(models, key=lambda model: model.get_human_str()))
    if len(models) > 4:
        return pair_order(dialog_str, models)
    return normal_order(dialog_str, models)
