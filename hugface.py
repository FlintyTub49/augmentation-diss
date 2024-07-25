import random
from copy import deepcopy

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

def generate_completions(triples, num_completion):
    '''
    Generate all possible completions for a given subject of a triple
    Args:
        triples: all the triples stored in a list format
        num_completion: the total number of continuations to generate for a particular subject
    Returns:
        selected_completions: all possible completions for a particular triple
    '''
    # Remove redundant subjects
    subjects = set(triple[0] for triple in triples)

    # Storing all the completions in a list to sample from and add to each subject
    all_completions = [f" {predicate} {obj}" for _, predicate, obj in triples]
    completions = {subject: [] for subject in subjects}
    
    # Selecting a fixed number of random completions for each subject
    selected_completions = {}
    for subject, _ in completions.items():
        selected_completions[subject] = random.sample(all_completions, len(all_completions))
    
    return selected_completions

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

def compute_likelihood(prompt, completions, batch_size=5):
    '''
    Compute the likelihood of a list of continuations for a particular subject
    Args:
        prompt: the subject for which to compute the likelihood
        completions: a list of possible completions which need to be scored
        batch_size: batch size for the computation
    Returns:
        likelihood: the likelihood for each continuation for the subject
    '''
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Iterate over completions
    likelihoods = []
    # ------------------------------ Batchwise ------------------------------ #
    for i in range(0, len(completions), batch_size):
        # Selecting a particular batch
        comp = completions[i : i + batch_size]
        
        # Tokenize all the completions in the particular batch and combine them with input
        completion_tokens = [tokenizer.encode(completion, return_tensors='pt').to(device) 
                             for completion in comp]
        input_tokens = [torch.cat((prompt_tokens, ct), dim=1) for ct in completion_tokens]
            
        padded_tokens = []
        for token in input_tokens:
            # Compute the padding length and pad with 0s
            padding_length = 100 - token.size(1)
            padded_tokens.append(F.pad(token, (0, padding_length), value=0))
            
        # input_tokens = torch.cat(padded_tokens, dim=0).to(device)
        input_tokens = torch.stack(padded_tokens).to(device)
        print(input_tokens.size())
        
        # Get model outputs and calculate the loss (negative log-likelihood)
        with torch.inference_mode():
            outputs = model(input_tokens, labels=input_tokens)
        losses = outputs.loss
        batch_likelihoods = -losses.cpu().numpy()
        likelihoods.append(batch_likelihoods)

    # -------------------------------- Online ------------------------------- #
    # for completion in completions:
    #     # Tokenize all the completions in the particular batch
    #     completion_tokens = tokenizer.encode(completion, return_tensors='pt').to(device)
        
    #     # Combine prompt and completion tokens
    #     input_tokens = torch.cat((prompt_tokens, completion_tokens), dim=1)
        
    #     # Combine prompt and completion tokens
    #     input_tokens = torch.cat((prompt_tokens, completion_tokens), dim=1).to(device)

    #     # Get model outputs and calculate the loss (negative log-likelihood)
    #     with torch.inference_mode():
    #         outputs = model(input_tokens, labels=input_tokens)
    #     loss = outputs.loss.item()
    #     likelihoods.append(-loss)

    return likelihoods

def compute_likelihood_icl(prompt, completions, batch_size=5):
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
    # Iterate over completions
    likelihoods = []
    # -------------------------------- Online ------------------------------- #
    for completion in completions:
        # Combine the prompt and the completion to create the model input
        model_input = prompt + completion
        input_tokens = tokenizer.encode(model_input, return_tensors='pt').to(device)

        # Get model outputs and calculate the loss (negative log-likelihood)
        with torch.inference_mode():
            outputs = model(input_tokens, labels=input_tokens)
        loss = outputs.loss.item()
        likelihoods.append(-loss)

    return likelihoods

def compute_likelihood_batched(prompt, completions, batch_size = 5):
    '''
    Compute the likelihood of a list of continuations for a particular subject in batches.
    
    Args:
        prompt: the subject for which to compute the likelihood
        completions: a list of possible completions which need to be scored
        model_name: the name of the model to be used
        batch_size: batch size for the computation
        
    Returns:
        likelihoods: a list of likelihoods for each continuation for the subject
    '''
    # Setting the padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    likelihoods = []
    # -------------- Batchwise calculation of score for triples ------------- #
    for i in range(0, len(completions), batch_size):
        batch_completions = completions[i:i + batch_size]
        
        # Tokenize all the completions in the batch
        batch_inputs = [prompt + completion for completion in batch_completions]
        input_tokens = tokenizer(batch_inputs, return_tensors='pt', padding=True).to(device)

        # Get model outputs and calculate the loss (negative log-likelihood)
        with torch.inference_mode():
            outputs = model(input_tokens['input_ids'], labels=input_tokens['input_ids'])
        
        # Collect the likelihoods for the batch
        batch_losses = outputs.loss.tolist()
        print(batch_losses)
        likelihoods.extend([-loss for loss in batch_losses])

    return likelihoods

def main():
    '''
    Main function to facilitate the running of the code
    Args:
        None
    Returns:
        None
    '''
    # Specify all the parameters for the code
    file_path = 'text_files/check.txt'
    num_completions = 20

    # Read the triples and generate the possible completions for each subject
    triples = read_triples(file_path)

    # ------------------ Scoring Each Possible Continuation ----------------- #
    # selected_completions = generate_completions(triples, num_completions)
    # for subject, completions in selected_completions.items():
    #     likelihoods = compute_likelihood(subject, completions)
    #     for i, likelihood in enumerate(likelihoods):
    #         print(f"> {subject} | {completions[i]} : {likelihood:.4f}")
            
    # ------------------------- In-context Learning ------------------------- #
    # Select random number of triples to serve as context
    joined = [' '.join(i) for i in triples]
    joined = random.sample(joined, 30)
    prompt = '\n'.join(joined)
    
    # Shuffle the available triples and send them as completions
    completions = generate_new_triples(deepcopy(triples))
    completions = [' '.join(i) for i in completions]
    likelihoods = compute_likelihood_icl(prompt, completions)
    
    for i, likelihood in enumerate(likelihoods):
        print(f">{completions[i]} : {likelihood:.4f}")
    

if __name__ == "__main__":
    # Making sure code runs on GPU
    device = 'mps'
    model_name = "gpt2"

    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model.eval()
    model.to(device)
    
    main()