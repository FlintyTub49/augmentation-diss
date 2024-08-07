import random
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def read_triples(file_path):
    '''
    Read a file from the given file path
    Args:
        file_path: path to the file
    Returns:
        None
    '''
    # Read the file and store the triples
    with open(file_path, 'r') as file:
        triples = [line.strip().split('\t') for line in file]
    return triples

def generate_new_triples(triples):
    '''Generate new artificial triples from the KG
    Args:
        triples: a list of triples from the original KG
    Returns:
        triples: the artificially generated triples
    '''
    predicate = [trip[1] for trip in triples]
    object = [trip[2] for trip in triples]

    # Shuffle each of them
    random.shuffle(predicate)
    random.shuffle(object)

    # Recreate artificial triples
    for i, trip in enumerate(triples):
        trip[1] = predicate[i]
        trip[2] = object[i]
        
    return triples

def triples_to_prompt(triples, size = 30):
    '''
    Convert a list of triples sampled to a specified size
    Args:
        triples: a list of triples to be converted to prompt
        size: random triples to be included in final prompt
    Returns:
        sampled: triples converted to striong sampled
    '''
    sampled = [' '.join(i) for i in triples]
    sampled = random.sample(sampled, min(size, len(triples)))
    sampled = '\n'.join(sampled)
    return sampled[:1024]

def compute_likelihood_icl(prompt, completion, batch_size=5):
    '''
    Compute the likelihood of a list of continuations for a particular subject
    Args:
        prompt: the subject for which to compute the likelihood
        completions: a list of possible completions which need to be scored
        model_name: the name of the model to be used
        batch_size: batch size for the computation
    Returns:
        likelihood: the likelihood for each continuation for the subject
    '''
    # Combine the prompt and the completion to create the model input
    model_input = prompt + completion
    input_tokens = tokenizer.encode(model_input, return_tensors='pt').to(device)

    # Get model outputs and calculate the loss (negative log-likelihood)
    with torch.inference_mode():
        outputs = model(input_tokens, labels=input_tokens)
    loss = outputs.loss.item()

    return -loss

def main(method = "spe"):
    '''
    Main function to facilitate the running of the code
    Args:
        None
    Returns:
        None
    '''

    # Read the triples and generate the possible completions for each subject
    triples = read_triples(file_path)
    
    # Shuffle the available triples and send them as completions
    completions = generate_new_triples(deepcopy(triples))
    completions = completions[:20000]
    
    # Iterate over the completions
    likelihoods = []
    for completion in tqdm(completions):
        # -------------------- Randomly Sampled KG as Context ------------------- #
        if method == 'ran':
            prompt = triples_to_prompt(triples)
            
        # --------------------- Specified Triples as Context -------------------- #
        elif method == 'spe':
            triples_to_select = []
            for i in triples:
                if i[0] == completion[0] or i[2] == completion[2]:
                    triples_to_select.append(i)

            prompt = triples_to_prompt(triples_to_select)

        completion = ' '.join(completion)
        likelihood = compute_likelihood_icl(prompt, completion)
        likelihoods.append(likelihood)
    
    # Sort the triples with highest likelihoods
    combined = list(zip(completions, likelihoods))
    
    with open('text_files/new_text/gene_gpt.txt', 'w') as file:
        for comp, likelihood in combined:
            file.write(f"{' '.join(comp)} : {likelihood:.4f}\n")
    
    with open('text_files/gene/gene_gpt.tsv', 'w') as file:
        for comp, _ in combined:
            file.write('\t'.join(comp) + '\n')
    
if __name__ == "__main__":
    # Making sure code runs on GPU
    device = 'mps'
    model_name = 'gpt2'
    method = 'spe'
    file_path = 'text_files/gene/train.tsv'

    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model.eval()
    model.to(device)
    
    main(method)