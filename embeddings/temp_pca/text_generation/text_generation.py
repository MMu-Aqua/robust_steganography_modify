import openai
import random
import re
import os


def clean_response(text):
    # Ensure text ends cleanly with a full sentence
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        return text[:match.end()].strip()
    else:
        return text.strip()


def generate_random_text(client, max_length):
    # Complex prompt designed to enforce diversity and mid-conversation starts
    system_prompt = (
        """You are an advanced AI model tasked with generating standalone text outputs that 
appear to be taken from the middle of a conversation or thought. Each output must 
differ in tone, content, structure, length, and style. Begin each text as if it is 
continuing a background context that the reader doesn't fully know, but make sure the 
text is engaging and meaningful on its own. Avoid repetition of ideas, themes, or phrasing 
across outputs. Examples might include a fragment of casual dialogue, an ongoing argument, 
a reflective thought, or a descriptive piece—but do not limit yourself to these styles. 
Ensure that all text is complete and coherent, does not start mid-sentence, and does not end abruptly. 
All text should be from the first person perspective and should not be things like 
dialogs or descriptions of the environment."""
    )

    try:
        # Generate a random response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate a unique text output."}
            ],
            max_tokens=max_length,
            temperature=1.0,  # Balanced variability
            top_p=1.0,
            frequency_penalty=0.6,                 # Discourage repetition moderately
            presence_penalty=0.5,
            stop=["\n"],
        )

        # Extract and clean the generated text
        text = response.choices[0].message.content.strip()
        text = clean_response(text)
        return text

    except Exception as e:
        return f"An error occurred: {e}"


def save_to_file(folder, texts, chunk_number):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Create the file path
    file_path = os.path.join(folder, f"chunk_{chunk_number}.txt")

    # Write texts to the file
    with open(file_path, "w", encoding="utf-8") as file:
        for text in texts:
            file.write(text + "\n\n")
    print(f"Saved chunk {chunk_number} with {len(texts)} texts to {file_path}")


if __name__ == "__main__":
    client = openai.OpenAI()
    output_folder = "generated_text_chunks"  # Folder to save files
    chunk_size = 10  # Number of generations per file
    total_generations = 20  # Total texts to generate

    all_texts = []
    chunk_number = 1

    for i in range(1, total_generations + 1):
        max_length = random.randint(20, 200)  # Randomize text length
        text = generate_random_text(client, max_length)
        all_texts.append(text)

        # Save to a file after every `chunk_size` generations
        if i % chunk_size == 0:
            save_to_file(output_folder, all_texts, chunk_number)
            all_texts = []  # Clear the buffer
            chunk_number += 1

    # Save any remaining texts in the buffer
    if all_texts:
        save_to_file(output_folder, all_texts, chunk_number)
