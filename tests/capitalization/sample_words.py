import json, random, string, sys

NUMBER_OF_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

INFILE = "words.txt"
OUTFILE = f"generated_words_{NUMBER_OF_WORDS}.json"
VALFILE = "val.json"

with open(INFILE, "r") as f:
    words = [line.strip() for line in f if line.strip()]

with open(VALFILE, "r") as vf:
    val_data = json.load(vf)
    val_words = set(item["input"].lower() for item in val_data)

all_five_letter_words = [
    word for word in words
    if len(word) == 5 and all(c.isalpha() for c in word) and word.lower() not in val_words
]
print(len(all_five_letter_words), "five-letter words found.")

five_letter_words_sample = random.sample(all_five_letter_words, NUMBER_OF_WORDS)

with open(OUTFILE, "w") as f:
    f.write("[\n")
    for i in range(NUMBER_OF_WORDS):
        word = five_letter_words_sample[i].lower()
        json.dump({"input": word, "output": word.capitalize()}, f)
        f.write(",\n" if i < NUMBER_OF_WORDS - 1 else "\n")
    f.write("]")
