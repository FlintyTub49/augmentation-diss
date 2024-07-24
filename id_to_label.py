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
    metadata = read_json('entity2wikidata.json')
    
    # Swap /m IDs with entity labels
    new_triples = []
    for i in triples:
        temp = i.split('\t')
        
        # Check if keys are available in the metadata
        if temp[0] in metadata.keys() and temp[2] in metadata.keys():
            temp[0] = metadata[temp[0]]['label']
            temp[2] = metadata[temp[2]]['label']
        
        # Making the new triple only if successfully converted
        if '/m' not in '\t'.join(temp):
            new_triples.append('\t'.join(temp))
        
    # Write the labelled triples to a new text file
    with open('text_files/new_sample.txt', 'w') as file:
        file.write('\n'.join(new_triples))
    return
    
if __name__ == "__main__":
    file_path = "text_files/train.txt"
    main(file_path)