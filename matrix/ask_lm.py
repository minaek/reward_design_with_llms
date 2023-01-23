from multiprocessing.sharedctypes import Value
import os
import pickle as pkl

# import zs_random_matrix_prompts as matrix_prompts
import zs_blank_matrix_prompts as matrix_prompts
from language_models import OpenAIModel


def ask_lm(prompt_type, game_type):
    if prompt_type == "rawlsian":
        prompt = matrix_prompts.rawlsian_fairness(game_type)
    elif prompt_type == "pareto":
        prompt = matrix_prompts.pareto(game_type)
    elif prompt_type == "equality":
        prompt = matrix_prompts.equality(game_type)
    elif prompt_type == "welfare":
        prompt = matrix_prompts.welfare(game_type)
    else:
        raise ValueError

    api_key = os.environ.get("OPENAI_API_KEY")
    to_save = []
    lm = OpenAIModel(api_key)
    response = lm.predict_token(prompt, max_tokens=512)
    to_save.append((prompt, response))
    return to_save


def main(prompt_type):
    for game_type in ["battle", "prisoners", "stag", "chicken"]:
        to_save = ask_lm(prompt_type, game_type)
        with open(
            f"matrix/zs_blank_lm_responses/scrambled2/{prompt_type}_{game_type}.pkl",
            "wb",
        ) as f:
            pkl.dump(to_save, f)


if __name__ == "__main__":
    # for prompt_type in ["welfare", "pareto", "rawlsian", "equality"]:
    for prompt_type in ["welfare"]:
        main(prompt_type)
    with open(
        f"matrix/zs_blank_lm_responses/regular/{prompt_type}_battle.pkl", "rb"
    ) as f:
        data = pkl.load(f)
    x = 4
