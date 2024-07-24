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
def conversation(args, content, instruct):
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
    
    # Specifying the model to chat with and sending the input and the instructions
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": instruct},
            {"role": "user", "content": content}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content
    
# Main Function to read input and present output
def main(args):
    '''
    Main function to facilitate running of all functions of the code 
    Args:
        args: All the model arguments
    Returns:
        None
    '''
    # If read from file is the command
    if args.type == 'file':
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
        print(f'Actual Input: {final_content}')

        repsonse = conversation(args, content=final_content, instruct=instruct)
        with open('output.txt', 'w') as file:
            file.write(repsonse)
    
    # For a continuous conversation with the LLM
    elif args.type == 'inline':
        while True:
            # Adding break for user
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'bye', 'end']:
                print("Exiting the conversation.")
                break
            
            # Sending response to model
            repsonse = conversation(content=user_input, instruct=None)
            print(f"Model: {repsonse}")
        
    return 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setting up simple arguments for the code")
    
    # Add arguments
    parser.add_argument('-t', '--type', type=str, default='file', choices=['file', 'inline'], 
                        help='How to interact with model')
    parser.add_argument('-p', '--path', type=str, default='text_files/sample.txt', 
                        help='Path to the necessary input file')
    parser.add_argument('-m', '--model', type=str, default='QuantFactory/Meta-Llama-3-8B-Instruct-GGUF', 
                        help='Model loaded onto LM Studio')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)