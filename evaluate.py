import argparse, re, csv
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

parser = argparse.ArgumentParser(description="Инструмент для оценки качества стихотворений на основе базы знаний")
parser.add_argument("--prompts", type=str, required=True, help="Путь к файлу *.csv со списком промптов для генерации стихотворений")
parser.add_argument("--num", type=int, required=True, help="Количество стихотворений для генерации (int)")
parser.add_argument("--temperature", type=float, required=True, help="Температура")
args = parser.parse_args()
prompt = args.prompt
num = args.num
temperature = args.temperature

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

# Подгрузка промптов с файла
with open('prompts/meta_prompt.txt') as f:
    system_prompt = f.read()

# Подгрузка базы данных из файла
with open('data/database.json') as f:
    database = f.read()

# Считывание промптов из файла
with open('data/prompts.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    prompts = [row for row in csvreader]

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
