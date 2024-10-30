import json

from src.final_environment.model_generation import Dialog
from src.final_environment.llm_heuristic import order_models
from src.final_environment.own_heuristic import NNModel
from src.logic.base.syntax import Formula


def print_formulas(*formulas: Formula, sort: bool = True, indent: int = 0) -> None:
    indentation = "  " * indent
    sep = f"{indentation}- "
    if sort:
        formulas = sorted(map(str, formulas), key=lambda x: (len(x), x))
    print(sep, end="")
    print(*formulas, sep=f"\n{sep}")


def main():
    with open("./data_hard.json", "r") as fp:
        raw_data = json.load(fp)

    dialogs = list(map(Dialog.from_dict, raw_data.get("annotations")))
    dialog = dialogs[0]
    final_models = dialog.get_models()
    models_list = [dialog.get_models(n) for n in range(len(dialog))]
    nn_model = NNModel()
    for dialog_number, dialog in enumerate(dialogs):
        if dialog_number > 0:
            print("\n" + ("----" * 20) + "\n")
        print(
            f"Dialog {dialog_number}: {'. '.join(s.sentence for s in dialog.sentences)}."
        )
        ordered_models = order_models(dialog)
        # ordered_models = nn_model.order_models(dialog)
        for model_number, model in enumerate(ordered_models):
            print(f"  - Model {model_number}: {model.get_human_str(True)}")
            # print_formulas(*reversed(model.reading_model.branch_literals), sort=False,  indent=2)


if __name__ == "__main__":
    main()
