import nltk
import re
import os
import math
import torch
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline


nltk.download('gutenberg')
nltk.download('punkt')


def preprocess_and_save():
    print("Preprocessing dataset...")
    text = gutenberg.raw('austen-emma.txt').lower()
    text = re.sub(r'[^a-zA-Z. ]+', '', text)
    sentences = sent_tokenize(text)
    os.makedirs('data', exist_ok=True)
    with open("data/processed.txt", "w") as f:
        for sent in sentences:
            f.write(sent.strip() + "\n")
    print("Saved processed data to data/processed.txt")


def load_model():
    print("Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, generator


def generate_text(generator, prompts):
    os.makedirs("outputs", exist_ok=True)
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        output = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        print(output)
        with open(f"outputs/output_{i+1}.txt", "w") as f:
            f.write(output)


def calculate_perplexity(model, tokenizer, text):
    print("\nCalculating perplexity...")

    device = torch.device("cpu")
    model.to(device)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    print(f"Perplexity: {perplexity}")
    return perplexity


if __name__ == "__main__":
    preprocess_and_save()
    model, tokenizer, generator = load_model()

    print("\n🔮 GPT-2 Text Generator")
    print("Type your prompt and press Enter to generate text.")
    print("Type 'exit' to stop.\n")

    while True:
        user_prompt = input("📝 Enter prompt: ")
        if user_prompt.lower() == 'exit':
            print("👋 Exiting program.")
            break

        output = generator(user_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        print(f"\n🧠 Generated Text:\n{output}\n")

        choice = input("🔍 Do you want to evaluate perplexity of this prompt? (y/n): ")
        if choice.lower() == 'y':
            calculate_perplexity(model, tokenizer, user_prompt)
