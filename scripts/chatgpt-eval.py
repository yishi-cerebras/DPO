import os
import openai
import asyncio
import json
from tqdm import tqdm
import os
from glob import glob
import argparse
import dotenv
import time
import re
import aiohttp
import shutil
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_KEY')
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2), before_sleep=print)
async def answer(content, session, semaphore):
    async with semaphore:
        await asyncio.sleep(1)
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": content}],
        }) as resp:
            response_json = await resp.json()
            return response_json["choices"][0]['message']["content"]

    

async def evaluate_generation(i, generation, prompt, directory_path, session, semaphore):
    policy_first = True
    responseA = generation['policy_response']
    responseB = generation['chosen_response']
    # change the order every time for more reliable result
    if i%2 == 0:
        responseA, responseB = responseB, responseA
        policy_first = False

    content = prompt.format(user_query=generation['prompt'], responseA=responseA, responseB=responseB)
    chatgpt_eval = await answer(content, session, semaphore)
    if chatgpt_eval == {}:
        raise EOFError
    chat_eval = chatgpt_eval

    gpt_response = {
        "prompt": content,
        "user_query": generation['prompt'],
        "policy_first": policy_first,
        "policy_response": generation['policy_response'],
        "chosen_response": generation['chosen_response'],
        "chatgpt_eval": chat_eval,
        "only_answer": chat_eval[chat_eval.rfind('More helpful'):]
    }

    with open(os.path.join(directory_path, f'gpt_response_{i}.json'), 'w') as f:
        json.dump(gpt_response, f)


def find_duplicates(directory):
    pattern = r'gpt_response_(\d+)\.json'
    numbers = []

    for filename in os.listdir(directory):
        match = re.search(pattern, filename)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


async def main(generations, prompt, directory_path):
    tasks = []
    gpt_responses = []
    duplicates = find_duplicates(directory_path)
    semaphore = asyncio.Semaphore(value=10)

    async with aiohttp.ClientSession() as session:
        for i, generation in enumerate(generations):
            if i not in duplicates:
                task = evaluate_generation(i, generation, prompt, directory_path, session, semaphore)
                tasks.append(task)

        await asyncio.gather(*tasks)
        return gpt_responses

def clean_up(directory_path):
    gpt_responses = []

    for name in os.listdir(directory_path):
        with open(os.path.join(directory_path, name), 'r') as f:
            gpt_response = json.loads(f.read())
            gpt_responses.append(gpt_response)

    shutil.rmtree(f'{directory_path}')
    with open(f'../generations/gpt_eval-{directory_path}-temp1.json', 'w') as f:
        json.dump(gpt_responses, f)

def print_results(directory_path):
    with open(f'../generations/gpt_eval-{directory_path}-temp1.json', 'r') as f:
        generations = json.load(f)

    count = len(generations)
    hit = 0

    for resp in generations:
        hit += (resp['policy_first'] and resp['only_answer'][-1] == 'A') or (not resp['policy_first'] and resp['only_answer'][-1] == 'B')
    print(f"Win rate: {hit/count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    name = args.name
    with open(f'../generations/{name}-temp1.json', 'r') as f:
        generations = json.load(f)
    with open('prompt.txt', 'r') as f:
        prompt = f.read()
    
    os.makedirs(directory_path, exist_ok=True)
    asyncio.run(main(generations, prompt, directory_path))
    
    clean_up(directory_path)
    print_results(directory_path)