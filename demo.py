import dashscope

dashscope.api_key = "sk-d9bcbecff5854fb09b684d060fc1d656"


completion = dashscope.Generation.call(
    model="qwen1.5-72b-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "怎么做蛋炒饭？"},
    ],
    temperature=1.0,
    max_tokens=100,
    top_p=0.9,
    result_format="message"
)


llm_response = completion.output.choices[0].message.content.strip()
print(llm_response)
