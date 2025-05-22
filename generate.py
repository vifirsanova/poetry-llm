import argparse, re
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

parser = argparse.ArgumentParser(description="Инструмент для генерации стихотворений на основе базы знаний")
parser.add_argument("--prompt", type=str, required=True, help="Описание стихотворения")
parser.add_argument("--temperature", type=float, required=True, help="Температура")
parser.add_argument("--lang", type=str, required=True, help="Язык: en/ru")
args = parser.parse_args()
prompt = args.prompt
temperature = args.temperature
lang = args.lang

model_id = "google/gemma-3-1b-it"

# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Определяем путь к файлу с промптом на нужном языке
prompt_path ='prompts/meta_prompt.txt' if lang == 'en' else 'prompts/meta_prompt_ru.txt' 

# Открываем файл с нужным промптом
with open(prompt_path) as f:
    system_prompt = f.read()

# Подгрузка базы данных из файла
with open('data/database.json') as f:
    database = f.read()

# Системные роли удобнее подгружать из отдельного файла
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt+database},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, temperature=temperature, top_k=50, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

# Добавляем парсинг ответа модели
pattern = r'<start_of_turn>(.*?)<end_of_turn>'
matches = re.findall(pattern, outputs[0], re.DOTALL)

print(matches[1])
