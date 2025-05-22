# poetry-llm
LLM-based tool for advanced poetry generation

## example usage

The model takes the following parameters:

- **temperature**: LLM hyperparameters
- **prompt**: user prompt for generating poetry in various genres and topics
- **lang**: base language for poetry generation

```bash
python3 generate.py --temperature 0.5 --prompt "Generate a haiku about spring" --lang "en"
```

Sample output:

```bash
model
Green shoots rise anew,\nSunlight warms the sleeping earth,\nLife bursts into bloom.\n
```

## evaluation script

