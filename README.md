# Poetry LLM

LLM-based tool for advanced poetry generation

## Example usage

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

## Evaluation script

Example usage:

```bash
python3 evaluate.py --prompts "./data/eval_data.csv" --num 5 --lang "en"
```

The script randomizes LLM temperature and returns *.csv with the following structure:

| prompt          | temperature        | result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Write a ballade | 0.9714108488780445 | model\nOkay, let’s generate a ballade based on your specifications. Here’s the output:\n\n**Ballade**\n\nThe wind does blow, a mournful sound,\\nAcross the moor, where shadows creep,\\nAnd silver frost, a chilling bound,\\nUpon the stones, in silent sleep.\\n\\nBeneath the boughs of ancient trees,\\nWhere wildness reigns, a mournful plea,\\nAnd whispers carried on the breeze,\\nOf lost love, eternally.\\n\\nThe river flows, a sluggish stream,\\nReflecting skies of gray and deep,\\nAnd lost to memory’s fading dream,\\nSecrets the silent water keep.\\n\\nYet, when the sun begins to fall,\\nAnd paints the clouds with hues of fire,\\nA sudden joy, a tender call,\\nIgnites within a heart’s desire.\\n\\nFor beauty lingers, ever bright,\\nAnd fades with time, in fading light,\\nAnd in my soul, a fervent flight,\\nOf memories, both warm and white.\\n\\nAnd I, a solitary soul,\\nTo dream of love, beyond control,\\nAnd seek a solace, pure and whole,\\nWithin the wild and misty role.\\n\\nSo let the wind forever blow,\\nAnd dance across the lonely height,\\nAnd let my spirit softly glow,\\nIn silent grace, a gentle light.\\n\\nLet it be a tale of love and woe,\\nOf fleeting moments, faint and deep,\\nAnd let my heart forever flow,\\nInto the mysteries where dreams sleep.\\n\\nAnd then, when the final curtain falls,\\nAnd darkness shrouds the lonely shore,\\nI’ll whisper softly, heed my calls,\\nAnd dream of love forevermore.\\n\\n---\n |
