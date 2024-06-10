import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional

import numpy as np
from gymnasium import spaces
from rich import print

from .config import (
    INDEX_TO_MOVE,
    LANG,
    META_INSTRUCTIONS,
    META_INSTRUCTIONS_WITH_LOWER,
    MOVES,
    NB_FRAME_WAIT,
    X_SIZE,
    Y_SIZE,
    logger,
)
from .observer import detect_position_from_color
from .llm import get_llm_client



class Robot:
    observations: List[Optional[dict]] = None  # memory
    next_steps: List[int]  # action plan
    actions: dict  # actions of the agents during a step of the game
    # actions of the agents during the previous step of the game
    previous_actions: Dict[str, List[int]]
    reward: float  # reward of the agent

    action_space: spaces.Space
    character: Optional[str] = None  # character name
    side: int  # side of the stage where playing: 0 = left, 1 = right
    current_direction: Literal["Left", "Right"]  # current direction facing
    sleepy: Optional[bool] = False  # if the robot is sleepy
    only_punch: Optional[bool] = False  # if the robot only punch

    model: str  # model of the robot
    super_bar_own: int
    player_nb: int  # player number

    def __init__(
        self,
        action_space: spaces.Space,
        character: str,
        side: int,
        character_color: list,
        ennemy_color: list,
        sleepy: bool = False,
        only_punch: bool = False,
        model: str = "mistral:mistral-large-latest",
        player_nb: int = 0,  # 0 means not specified
    ):
        self.action_space = action_space
        self.character = character
        if side == 0:
            self.current_direction = "Right"
        elif side == 1:
            self.current_direction = "Left"

        self.observations = []
        self.next_steps = []
        self.character_color = character_color
        self.ennemy_color = ennemy_color
        self.side = side
        self.sleepy = sleepy
        self.only_punch = only_punch
        self.model = model
        self.previous_actions = defaultdict(list)
        self.actions = {}
        self.player_nb = player_nb

    def act(self) -> int:
        """
        At each game frame, we execute the first action in the list of next steps.

        An action is an integer from 0 to 18, where 0 is no action.

        See the MOVES dictionary for the mapping of actions to moves.
        """
        if not self.next_steps or len(self.next_steps) == 0:
            return 0  # No move

        if self.sleepy:
            return 0

        if self.only_punch:
            # Do a Hadouken
            if self.current_direction == "Right":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Right+Down"],
                        MOVES["Right"],
                        MOVES["High Punch"],
                    ]
                )
            elif self.current_direction == "Left":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Down+Left"],
                        MOVES["Left"],
                        MOVES["High Punch"],
                    ]
                )

        next_step = self.next_steps.pop(0)

        return next_step

    def plan(self) -> None:
        """
        The robot will plan its next steps by calling this method.

        In SF3, moves are based on combos, which are list of actions that must be executed in a sequence.

        Moves of Ken
        https://www.eventhubs.com/guides/2009/may/11/ken-street-fighter-3-third-strike-character-guide/

        Moves of Ryu
        https://www.eventhubs.com/guides/2008/may/09/ryu-street-fighter-3-third-strike-character-guide/
        """

        # If we already have a next step, we don't need to plan
        if len(self.next_steps) > 0:
            return

        # Call the LLM to get the next steps
        next_steps_from_llm = self.get_moves_from_llm()
        next_buttons_to_press = [
            button
            for combo in next_steps_from_llm
            for button in META_INSTRUCTIONS_WITH_LOWER[combo][
                self.current_direction.lower()
            ]
            # We add a wait time after each button press
            + [0] * NB_FRAME_WAIT
        ]
        self.next_steps.extend(next_buttons_to_press)

    def observe(self, observation: dict, actions: dict, reward: float):
        """
        The robot will observe the environment by calling this method.

        The latest observations are at the end of the list.
        """

        # detect the position of characters and ennemy based on color
        observation["character_position"] = detect_position_from_color(
            observation, self.character_color
        )
        observation["ennemy_position"] = detect_position_from_color(
            observation, self.ennemy_color
        )

        self.observations.append(observation)
        # we delete the oldest observation if we have more than 10 observations
        if len(self.observations) > 10:
            self.observations.pop(0)

        self.reward = reward

        if actions.get("agent_0") is not None and actions.get("agent_0") != 0:
            self.previous_actions["agent_0"].append(actions["agent_0"])
        if actions.get("agent_1") is not None and actions.get("agent_1") != 0:
            self.previous_actions["agent_1"].append(actions["agent_1"])

        for key, value in actions.items():
            if len(self.previous_actions[key]) > 10:
                self.previous_actions[key].pop(0)

        # Keep track of the current direction by checking the position of the character
        # and the ennemy
        character_position = observation.get("character_position")
        ennemy_position = observation.get("ennemy_position")
        if (
            character_position is not None
            and ennemy_position is not None
            and len(character_position) == 2
            and len(ennemy_position) == 2
        ):
            if character_position[0] < ennemy_position[0]:
                self.current_direction = "Right"
            else:
                self.current_direction = "Left"

    def context_prompt(self) -> str:
        """
        Return a str of the context

        "The observation for you is Left"
        "The observation for the opponent is Left+Up"
        "The action history is Up"
        """

        # Create the position prompt
        side = self.side
        obs_own = self.observations[-1]["character_position"]
        obs_opp = self.observations[-1]["ennemy_position"]
        super_bar_own = self.observations[-1]["P" + str(side + 1)]["super_bar"][0]

        if obs_own is not None and obs_opp is not None:
            relative_position = np.array(obs_own) - np.array(obs_opp)
            normalized_relative_position = [
                relative_position[0] / X_SIZE,
                relative_position[1] / Y_SIZE,
            ]
        else:
            normalized_relative_position = [0.3, 0]

        position_prompt = ""
        if abs(normalized_relative_position[0]) > 0.1:
            if LANG == "ZH":
                position_prompt += "你离对手很远。靠近对手。"
            else:
                position_prompt += (
                    "You are very far from the opponent. Move closer to the opponent."
                )
            if normalized_relative_position[0] < 0:
                if LANG == "ZH":
                    position_prompt += "你的对手在右边。"
                else:
                    position_prompt += "Your opponent is on the right."
            else:
                if LANG == "ZH":
                    position_prompt += "你的对手在左边。"
                else:
                    position_prompt += "Your opponent is on the left."

        else:
            if LANG == "ZH":
                position_prompt += "你离对手很近。你应该攻击他。"
            else:
                position_prompt += (
                    "You are close to the opponent. You should attack him."
                )

        power_prompt = ""
        if super_bar_own >= 30:
            if LANG == "ZH":
                power_prompt = "您现在可以使用强大的动作。强大动作的名称是：Megafireball, Super attack 2"
            else:
                power_prompt = "You can now use a powerfull move. The names of the powerful moves are: Megafireball, Super attack 2"
        if super_bar_own >= 120 or super_bar_own == 0:
            if LANG == "ZH":
                power_prompt = (
                    "你现在只能使用非常强大的动作。非常强大的动作的名称是：Super attack 3, Super attack 4"
                )
            else:
                power_prompt = "You can now only use very powerfull moves. The names of the very powerful moves are: Super attack 3, Super attack 4"
        # Create the last action prompt
        last_action_prompt = ""
        if len(self.previous_actions.keys()) >= 0:
            act_own_list = self.previous_actions["agent_" + str(side)]
            act_opp_list = self.previous_actions["agent_" + str(abs(1 - side))]

            if len(act_own_list) == 0:
                act_own = 0
            else:
                act_own = act_own_list[-1]
            if len(act_opp_list) == 0:
                act_opp = 0
            else:
                act_opp = act_opp_list[-1]

            str_act_own = INDEX_TO_MOVE[act_own]
            str_act_opp = INDEX_TO_MOVE[act_opp]

            if LANG == "ZH":
                last_action_prompt += (
                    f"你的最后一个动作是 {str_act_own}. 对手的最后一个动作是 {str_act_opp}."
                )
            else:
                last_action_prompt += f"Your last action was {str_act_own}. The opponent's last action was {str_act_opp}."

        reward = self.reward

        # Create the score prompt
        score_prompt = ""
        if reward > 0:
            if LANG == "ZH":
                score_prompt += "你赢了。继续攻击对手。"
            else:
                score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            if LANG == "ZH":
                score_prompt += "你输了。继续攻击对手，但不要被击中。"
            else:
                score_prompt += (
                    "You are losing. Continue to attack the opponent but don't get hit."
                )

        # Assemble everything
        context = f"""{position_prompt}
{power_prompt}
{last_action_prompt}
Your current score is {reward}. {score_prompt}
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
"""
        context_zh = f"""{position_prompt}
{power_prompt}
{last_action_prompt}
你当前的分数是 {reward}。 {score_prompt}
为了增加你的分数，向对手移动并攻击对手。为了防止你的分数下降，不要被对手击中。
"""

        if LANG == "ZH":
            context = context_zh
        return context

    def get_moves_from_llm(
        self,
    ) -> List[str]:
        """
        Get a list of moves from the language model.
        """

        # Filter the moves that are not in the list of moves
        invalid_moves = []
        valid_moves = []

        # If we are in the test environment, we don't want to call the LLM
        if os.getenv("DISABLE_LLM", "False") == "True":
            # Choose a random int from the list of moves
            logger.debug("DISABLE_LLM is True, returning a random move")
            return [random.choice(list(MOVES.values()))]

        while len(valid_moves) == 0:
            llm_response = self.call_llm()

            # The response is a bullet point list of moves. Use regex
            matches = re.findall(r"- ([\w ]+)", llm_response)
            moves = ["".join(match) for match in matches]
            invalid_moves = []
            valid_moves = []
            for move in moves:
                cleaned_move_name = move.strip().lower()
                if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                    if self.player_nb == 1:
                        print(
                            f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                        )
                    elif self.player_nb == 2:
                        print(
                            f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                        )
                    valid_moves.append(cleaned_move_name)
                else:
                    logger.debug(f"Invalid completion: {move}")
                    logger.debug(f"Cleaned move name: {cleaned_move_name}")
                    invalid_moves.append(move)

            if len(invalid_moves) > 1:
                logger.warning(f"Many invalid moves: {invalid_moves}")

        logger.debug(f"Next moves: {valid_moves}")
        return valid_moves

    def call_llm(
        self,
        temperature: float = 0.95,
        max_tokens: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Make an API call to the language model.

        Edit this method to change the behavior of the robot!
        """
        client = get_llm_client(self.model)

        # Generate the prompts
        move_list = "- " + "\n- ".join([move for move in META_INSTRUCTIONS])
        system_prompt = f"""You are the best and most aggressive Street Fighter III 3rd strike player in the world.
Your character is {self.character}. Your goal is to beat the other opponent. You respond with a bullet point list of moves.
{self.context_prompt()}
The moves you can use are:
{move_list}
----
Reply with a bullet point list of moves. The format should be: `- <name of the move>` separated by a new line.
Example if the opponent is close:
- Move closer
- Medium Punch

Example if the opponent is far:
- Fireball
- Move closer"""

        system_prompt_zh = f"""你是世界上最好、最具侵略性的街头霸王 III 3rd Strike 玩家。
你的角色是 {self.character}。 你的目标是击败另一个对手。你用一个动作列表来做出响应。
{self.context_prompt()}
你可以使用的动作如下：
{move_list}
----
回复一个动作列表。格式应为：“- <动作名称>，用换行分隔。
比如，如果对手离你比较近，你就输出：
- Move closer
- Medium Punch

再比如，如果对手离你比较远，你就输出：
- Fireball
- Move closer"""

        user_init = "Your next moves are:"
        user_init_zh = "你接下来的动作要点是（直接输出英文的动作列表，不要任何解释）："
        if LANG == "ZH":
            system_prompt = system_prompt_zh
            user_init = user_init_zh
        start_time = time.time()
        llm_response = client.call(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_init},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        logger.debug(f"LLM call to {self.model}: {system_prompt}")
        logger.debug(f"LLM call to {self.model}: {time.time() - start_time}s")
        logger.debug(f"\n\n[yellow] LLM {self.model} generate:\n{llm_response}\n\n")
        return llm_response
