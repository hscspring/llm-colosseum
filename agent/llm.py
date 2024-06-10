import os
from typing import List, Dict

import dashscope
from zhipuai import ZhipuAI



ZHIPU_KEY = os.getenv("ZHIPU_KEY")
DASHSCOPE_KEY = os.getenv("DASHSCOPE_KEY")
if not DASHSCOPE_KEY or not ZHIPU_KEY:
    raise ValueError("请指定DASHSCOPE_KEY和ZHIPU_KEY的环境变量")


MODELS = {
    "GLM": {
        "glm-4",
        "glm-3-turbo",
    },
    "QWEN": {
        "qwen-plus",
        "qwen-turbo",
        "qwen1.5-72b-chat",
        "qwen1.5-14b-chat",
        "qwen1.5-7b-chat",
        "qwen1.5-1.8b-chat",
    },
}




class QwenClient:
    def __init__(self):
        dashscope.api_key = DASHSCOPE_KEY

    def call(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.95,
        max_tokens: int = 20,
        top_p: float = 0.9,
    ) -> str:
        response = None
        count = 0
        while response is None or response.status_code != 200:
            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                result_format="message",
            )
            count += 1
            if count > 1:
                time.sleep(3)
        llm_response = response.output.choices[0].message.content.strip()
        return llm_response


class GlmClient:
    def __init__(self):
        self.client = ZhipuAI(api_key=ZHIPU_KEY)

    def call(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.95,
        max_tokens: int = 20,
        top_p: float = 0.9,
    ) -> str:
        response = None
        count = 0
        while response is None or response.choices is None:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )
            count += 1
            if count > 1:
                time.sleep(3)
        llm_response = response.choices[0].message.content.strip()
        return llm_response



def get_llm_client(mid: str):
    if "glm" in mid:
        return GlmClient()
    elif "qwen" in mid:
        return QwenClient()



def generate_random_model(glm: bool = False, qwen: bool = True):
    models_available = []

    for model, models in MODELS.items():
        if qwen and model == "GLM":
            models_available.extend(models)
        if glm and model == "QWEN":
            models_available.extend(models)

    random.seed()
    # Generate a pair of random two models
    random_model = random.choice(models_available)

    return random_model



