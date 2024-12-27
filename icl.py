from omegaconf import OmegaConf
import argparse
import sys
import aim
import ollama
from typing import List, Tuple, Dict, TypeAlias
from accuracy import token_lvl_accuracy, seq_lvl_accuracy
from tqdm import tqdm
import random

EXP1_PATH = "experiment1/"
EXP3_PATH = "experiment3/"
EXP1_SPLITS = [1, 2, 4, 8, 16, 32, 64]
MODELS = {
    "llama8b": "llama3.1:8b-instruct-q4_0",
    "mistral7b": "mistral:7b-instruct-q4_0",
    "llama3b": "llama3.2:3b-instruct-q4_0",
}
PROMPTS = {"normal": "", "cot": ""}
ACTIONS2TOKEN = {
    "I_TURN_RIGHT": "right",
    "I_TURN_LEFT": "left",
    "I_JUMP": "jump",
    "I_RUN": "run",
    "I_WALK": "walk",
    "I_LOOK": "look",
}
ACTIONS2TOKEN_NULL = {
    "I_TURN_RIGHT": "cat",
    "I_TURN_LEFT": "dog",
    "I_JUMP": "pea",
    "I_RUN": "tea",
    "I_WALK": "bug",
    "I_LOOK": "fly",
}
TOKEN2ACTIONS = {
    "right": "I_TURN_RIGHT",
    "left": "I_TURN_LEFT",
    "jump": "I_JUMP",
    "run": "I_RUN",
    "walk": "I_WALK",
    "look": "I_LOOK",
}
TOKEN_NULL2ACTIONS = {
    "cat": "I_TURN_RIGHT",
    "dog": "I_TURN_LEFT",
    "pea": "I_JUMP",
    "tea": "I_RUN",
    "bug": "I_WALK",
    "fly": "I_LOOK",
}
MAX_TESTS = 100

Data: TypeAlias = List[Tuple[str, str]]
SplitData: TypeAlias = Dict[str, Data]


def split_commands_actions(s: str) -> tuple:
    _in = s.split(" OUT:")[0][4:]
    _out = s.split(" OUT:")[1]
    return (_in, _out.strip())


def token_null_to_actions(tokens):
    return " ".join(
        [
            TOKEN_NULL2ACTIONS[token] if token in TOKEN_NULL2ACTIONS else ""
            for token in tokens
        ]
    )


def token_to_actions(tokens):
    return " ".join(
        [TOKEN2ACTIONS[token] if token in TOKEN2ACTIONS else "" for token in tokens]
    )


def actions_to_token(actions):
    return " ".join([ACTIONS2TOKEN[action] for action in actions])


def actions_to_token_null(actions):
    return " ".join([ACTIONS2TOKEN_NULL[action] for action in actions])


def parse_data_exp1(exp_split) -> Tuple[list, list]:
    train_data = []
    test_data = []

    train_split = open(
        f"{EXP1_PATH}/tasks_train_simple_p{exp_split}.txt", "r"
    ).readlines()
    test_split = open(
        f"{EXP1_PATH}/tasks_test_simple_p{exp_split}.txt", "r"
    ).readlines()
    train_data = [split_commands_actions(line) for line in train_split]
    test_data = [split_commands_actions(line) for line in test_split[:MAX_TESTS]]

    print(
        f"Split {exp_split}%: train size {len(train_data)}, test size {len(test_data)}"
    )
    return train_data, test_data


def experiment1(run, train_data: Data, test_data: Data):
    # get prompts from train data and run inference on test data
    prefix = (
        "Translate the following Commands into Actions, providing only the answer.\n"
    )

    # randomly shuffle the training data and pick some examples based on the split
    num_shots = int(run["hparams"]["exp_split"])
    train_examples = train_data[:num_shots]
    examples = ""
    preprocess_fn = None
    postprocess_fn = None

    # the code looks uglier, but this way I check only once and not in each loop
    if run["hparams"]["prompt"] == "token_changed":
        preprocess_fn = actions_to_token
        postprocess_fn = token_to_actions

    elif run["hparams"]["prompt"] == "token_changed_null":
        preprocess_fn = actions_to_token_null
        postprocess_fn = token_null_to_actions

    else:
        preprocess_fn = lambda x: x
        postprocess_fn = lambda x: x

    # elif run["hparams"]["prompt"] == "cot":
    #     # TODO manually add a cot prompt
    #     pass

    examples = "\n".join(
        [
            f"{prefix}Commands: {sample[0]}\nActions: {preprocess_fn(sample[1])}\n"
            for sample in train_examples
        ]
    )

    for step in (pbar := tqdm(range(0, len(test_data)), total=len(test_data))):
        test_input, test_output = test_data[step]
        full_prompt = f"{examples}Commands: {test_input}\nActions: "
        res = ollama.generate(
            model=run["hparams"]["model"], prompt=full_prompt, options={"num_ctx": 4600}
        )
        prediction = res.response
        tok_acc = token_lvl_accuracy(test_output, postprocess_fn(prediction))
        seq_acc = seq_lvl_accuracy(test_output, postprocess_fn(prediction))
        run.track(tok_acc, name="token_lvl_acc")
        run.track(seq_acc, name="sequence_lvl_acc")
        pbar.set_description(
            f"Token lvl acc: {tok_acc:.3f}, Sequence lvl acc: {seq_acc:.3f}"
        )


def main():
    parser = argparse.ArgumentParser("ICL inference")
    parser.add_argument(
        "--experiment",
        dest="experiment",
        type=int,
        default=1,
        help="The expriment to run. Can be `1` or `2`",
    )
    parser.add_argument(
        "--split",
        dest="data_split",
        default="1",
        help="Split of data. Can be any of `1`, `2`, `4`, `8`, `16`, `32`, `64`",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="llama3b",
        help="Model to use. Can be `llama3b`, `llama8b`, or `mistral7b`",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt",
        default="normal",
        help="Prompt to use. Can be `normal`, `token_changed`, or `cot` (Chain of Thought)",
    )
    args = parser.parse_args()
    config = OmegaConf.create(vars(args))

    run = aim.Run()
    run["hparams"] = {
        "model": MODELS[config.model],
        "experiment": config.experiment,
        "exp_split": config.data_split,
        "prompt": config.prompt,
    }

    if config.experiment == 1:
        exp_split = config.data_split
        train_data, test_data = parse_data_exp1(exp_split)
        experiment1(run, train_data, test_data)

    elif config.experiment == 3:
        # train_data, test_data = parse_data_exp1(config.split)
        # experiment3(run, train_data, test_data)
        pass

    else:
        print("Invalid experiment number")
        sys.exit(-1)


if __name__ == "__main__":
    # argv = sys.argv
    main()
