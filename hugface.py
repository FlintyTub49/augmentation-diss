import random
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

def compute_likelihood(prompt, completions, model_name, batch_size=5):
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
    # Making sure code runs on GPU
    device = 'mps'
    
    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    model.to(device)

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
        with torch.no_grad():
            outputs = model(input_tokens, labels=input_tokens)
        losses = outputs.loss
        batch_likelihoods = -losses.cpu().numpy()
        likelihoods.append(batch_likelihoods)

    # ------------------------------ Stochastic ----------------------------- #
    # for completion in completions:
    #     # Tokenize all the completions in the particular batch
    #     completion_tokens = tokenizer.encode(completion, return_tensors='pt').to(device)
        
    #     # Combine prompt and completion tokens
    #     input_tokens = torch.cat((prompt_tokens, completion_tokens), dim=1)
        
    #     # Combine prompt and completion tokens
    #     input_tokens = torch.cat((prompt_tokens, completion_tokens), dim=1).to(device)

    #     # Get model outputs and calculate the loss (negative log-likelihood)
    #     with torch.no_grad():
    #         outputs = model(input_tokens, labels=input_tokens)
    #     loss = outputs.loss.item()
    #     likelihoods.append(-loss)

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
    file_path = 'check.txt'
    model_name = 'gpt2'
    num_completions = 20

    # Read the triples and generate the possible completions for each subject
    triples = read_triples(file_path)
    selected_completions = generate_completions(triples, num_completions)
    # print(selected_completions)

    # Compute the possible likelihood for each subject and continuation
    for subject, completions in selected_completions.items():
        likelihoods = compute_likelihood(subject, completions, model_name)
        for i, likelihood in enumerate(likelihoods):
            print(f"> {subject} | {completions[i]} : {likelihood:.4f}") #TODO: Save maximum likelihood continuation instead of printing all

if __name__ == "__main__":
    main()