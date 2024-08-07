import random

# Function to read the file
def read_file(file_path):
    triples = []
    try:
        with open(file_path, "r") as file:
            for i in file:
                triples.append(i.strip())
        return triples
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
# Main function
def main(org, aug, sample, model):
    original = read_file(org)
    new_add = read_file(aug)
    to_add = random.sample(new_add, sample)
    
    final = to_add + original
    with open(f'text_files/fb15k/fb15k_{sample}_{model}.tsv', 'w') as file:
        file.write('\n'.join(final))
        
if __name__ == "__main__":
    org = "text_files/fb15k/train.tsv"
    
    for m in ['gpt', 'gemma']:
        aug = f"text_files/fb15k/fb15k_{m}.tsv"
        for i in [15, 50, 150]:
            main(org, aug, sample = i, model = m)