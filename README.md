# README

Install: https://github.com/OpenGenerativeAI/llm-colosseum

run:

```bash
# cp .env.example .env
# add your key
source .env
# run
LAN=zh LOG=debug diambra -r ~/.diambra/roms --env.image diambra/engine:latest run python3.11 script.py  glm-3-turbo qwen-turbo
```

