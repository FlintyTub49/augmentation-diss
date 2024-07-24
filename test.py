def compute_likelihood_batched(prompt, completions, model_name, batch_size):
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
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)

    likelihoods = []
    # Iterate over completions in batches
    for i in range(0, len(completions), batch_size):
        batch_completions = completions[i:i + batch_size]
        
        # Tokenize all the completions in the batch
        batch_inputs = [prompt + completion for completion in batch_completions]
        input_tokens = tokenizer.batch_encode_plus(batch_inputs, return_tensors='pt', padding=True).to(device)

        # Get model outputs and calculate the loss (negative log-likelihood)
        with torch.inference_mode():
            outputs = model(input_tokens['input_ids'], labels=input_tokens['input_ids'])
        
        # Collect the likelihoods for the batch
        batch_losses = outputs.loss.tolist()
        likelihoods.extend([-loss for loss in batch_losses])

    return likelihoods