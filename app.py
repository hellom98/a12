import os
import torch
import gradio as gr
from train import GPT, GPTConfig
import tiktoken

# Initialize tokenizer
tokenizer = tiktoken.get_encoding('gpt2')

# Load the model
def load_model():
    model_path = 'model_checkpoints/gpt_model.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Please train the model first.")
    
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Text generation function
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    try:
        model = load_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        
        generated_text = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output
                logits = model(input_ids)[0]
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices[0, idx_next[0]]
                
                # Append to generated sequence
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
                
                # Convert token to text
                next_token_text = tokenizer.decode([next_token.item()])
                generated_text.append(next_token_text)
                
                # Stop if we generate a newline
                if next_token_text == '\n\n':
                    break
        
        # Combine prompt and generated text
        full_text = prompt + ''.join(generated_text)
        return full_text
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Enter your prompt"),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(lines=5, label="Generated Text"),
    title="GPT Text Generator",
    description="Enter a prompt and the model will generate text based on it.",
    examples=[
        ["The quick brown fox", 100, 0.7, 50],
        ["Once upon a time", 150, 0.8, 40],
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 