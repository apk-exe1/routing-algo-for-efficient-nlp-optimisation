import random
import os

def generate_kannada_word(length=5):
    # Base Kannada consonants and vowels
    consonants = [chr(i) for i in range(0x0C95, 0x0CB9 + 1)]
    vowels = [chr(i) for i in range(0x0CBE, 0x0CCC + 1)]
    
    word = ""
    for _ in range(length):
        word += random.choice(consonants)
        if random.random() > 0.3:  # 70% chance to have a vowel modifier
            word += random.choice(vowels)
    return word

def generate_dictionary(num_words=100000, filename="data/sample_dictionary_100k.txt"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    words = set()
    print(f"Generating {num_words} Kannada words...")
    while len(words) < num_words:
        # random length between 3 and 8 syllables
        length = random.randint(3, 8)
        word = generate_kannada_word(length)
        words.add(word)
    
    with open(filename, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")
    print(f"Dictionary saved to {filename}")

if __name__ == "__main__":
    generate_dictionary()
