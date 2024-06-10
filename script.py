import os
import sys


from agent.llm import MODELS, generate_random_model
from eval.game import Game, Player1, Player2


def main():
    all_models = MODELS["GLM"] | MODELS["QWEN"]
    p1 = p2 = "init"
    if len(sys.argv) < 2:
        while p1 == p2:
            p1 = generate_random_model(glm=True, qwen=True)
            p2 = generate_random_model(glm=True, qwen=True)
    else:
        p1 = sys.argv[1]
        p2 = sys.argv[2]

    assert p1 in all_models and p2 in all_models
    print(f"{p1} VS {p2}")

    game = Game(
        render=True,
        player_1=Player1(
            nickname=p1.split("-")[0],
            model=p1,
        ),
        player_2=Player2(
            nickname=p2.split("-")[0],
            model=p2,
        ),
    )
    return game.run()


if __name__ == "__main__":
    main()
