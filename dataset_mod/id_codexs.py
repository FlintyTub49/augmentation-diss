import json

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
    
# Function to read the metadata file
def read_json(file_path):
    with open(file_path) as file:
        return json.load(file)
    
# Main function
def main(file_path):
    # Read the source triples file and metadata
    triples = read_file(file_path)
    ent = read_json('entities.json')
    rel = read_json('relations.json')
    
    # Swap IDs with labels
    new_triples = []
    for i in triples:
        s, p, o = i.split('\t')
        
        # Check if keys are available in the metadata
        if s in ent.keys() and p in rel.keys() and o in ent.keys():
            s = ent[s]['label']
            p = rel[p]['label']
            o = ent[o]['label']
            
        new_triples.append('\t'.join([s, p, o]))
    
    # Write the labelled triples to a new text file
    with open('text_files/codex-s/train.tsv', 'w') as file:
        file.write('\n'.join(new_triples))
    return
    
if __name__ == "__main__":
    file_path = "train.txt"
    main(file_path)
    