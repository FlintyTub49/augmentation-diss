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
        print(temp)
        
        for j in metadata.keys():
            if metadata[j]['label'] == temp[0]: temp[0] = j
            if metadata[j]['label'] == temp[2]: temp[2] = j
                
        # Making the new triple only if it successfully converted back to ID
        if '/m' in temp[0] and '/m' in temp[2]:
            new_triples.append('\t'.join(temp))
        
    # Write the labelled triples to a new text file
    with open('text_files/new_input.txt', 'w') as file:
        file.write('\n'.join(new_triples))
    return
    
if __name__ == "__main__":
    file_path = "text_files/output.txt"
    main(file_path)