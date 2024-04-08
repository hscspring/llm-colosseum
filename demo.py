from agent.robot import GlmClient, QwenClient



glm = GlmClient()
qwen = QwenClient()


messages=[
    {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
    {"role": "user", "content": "我对太阳系的行星非常感兴趣，特别是土星。请提供关于土星的基本信息，包括其大小、组成、环系统和任何独特的天文现象。"},
]

r1 = glm.call(
    "glm-4",
    messages,
)
print(r1)
r2 = qwen.call(
    "qwen-max",
    messages,
)
print(r2)
