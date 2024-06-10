# README

Install: https://github.com/OpenGenerativeAI/llm-colosseum

run:

```bash
# cp .env.example .env
# add your key
source .env

# engine
docker run -it --rm --name engine -p 50051:50051 \
  -v $HOME/.diambra/credentials:/tmp/.diambra/credentials \
  -v $HOME/.diambra/roms:/opt/diambraArena/roms \
  docker.io/diambra/engine:latest

# run
LAN=zh LOG=debug DIAMBRA_ENVS=localhost:50051 python3.11 script.py  glm-3-turbo qwen-turbo
```

blog: https://yam.gift/2024/04/08/NLP/2024-04-08-LLM-Colosseum/
