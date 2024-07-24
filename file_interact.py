from openai import OpenAI
import argparse

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Read a file for the input
def read_file_content(file_path):
    '''
    Read a file from the given file path
    Args:
        file_path: path to the file
    Returns:
        None
    '''
    print(file_path)
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Have conversation with the model via the local server
def conversation(args, instruct, content):
    '''
    Have a conversation with the LLM model via the local server
    Args:
        args: All the model arguments
        content: The content to be sent be sent to the model
        instruct: Default instructions for the model
    Returns:
        responde of the model
    '''
    # Default instrutions
    if not instruct:
        instruct = "Always answer in one sentence"
        
    history = [
        {"role": "system", "content": instruct},
        {"role": "user", "content": content}
    ]
    
    while True:
        # Specifying the model to chat with and sending the input and the instructions
        completion = client.chat.completions.create(
            model=args.model,
            messages=history,
            temperature=0.7,
            stream=True
        )
        
        # Extracting the output from the model
        new_message = {"role": "assistant", "content": ""}
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content
                
        # Saving the system generated output to be appeneded to history
        model_output = new_message["content"]
        history.append(new_message)
        
        # Taking next input from user
        print()
        user_input = input("\nUser: ").lower()
        while True:
            # Finish the conversation if the keyword is triggered
            if user_input in ['exit', 'bye', 'end']:
                print("Exiting the conversation.")
                return
            
            # Save the previous model output and then continue taking input from the user
            elif 'save' in user_input and len(user_input.split()) == 2:
                with open(user_input.split()[1], 'w') as file:
                    file.write(model_output)
                user_input = input("User: ")
                continue
            
            # If a keyword is not triggered, send next input to the model
            else:
                history.append({"role": "user", "content": user_input})
                break
    
# Main Function to read input and present output
def main(args):
    '''
    Main function to facilitate running of all functions of the code 
    Args:
        args: All the model arguments
    Returns:
        None
    '''
    # Default file path
    content = read_file_content(args.path)
    
    # If no file received, keep asking for path
    if content is None:
        while content is None:
            print("No input recieved, provide path again.")
            path = input("Path: ")
            content = read_file_content(path)
            
    # Printing the actual input to the model
    instruct = 'exactly follow the prompt provided'
    final_content = read_file_content("instruct.txt") + '\n' + content
    print(f'Actual Input: {final_content}\n')

    # Send all the input to the model
    conversation(args, instruct=instruct, content=final_content)
        
    return 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setting up simple arguments for the code")
    
    # Add arguments
    parser.add_argument('-p', '--path', type=str, default='sample.txt', 
                        help='Path to the necessary input file')
    parser.add_argument('-m', '--model', type=str, default='QuantFactory/Meta-Llama-3-8B-Instruct-GGUF', 
                        help='Model loaded onto LM Studio')
    # parser.add_argument('-t', '--type', type=str, default='file', choices=['file', 'inline'], 
    #                     help='How to interact with model')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)