from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import check_cuda
import torch

def run_gt():
    check_cuda()
    # print(f"Generated text: " + generate_text())

def generate_text():
    # We're going to use the AutoTokenizer class to automatically download the tokenizer suitable for the model we're using
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", padding_side="left")
    # Create a quantization configuration object to use with the model
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4",
                                             # bnb_4bit_compute_dtype=torch.bfloat16
                                             bnb_4bit_compute_dtype=torch.float32 # better CPU compatibility
                                             )
    # Create a new instance of the LLM model in a variable using a pre-trained model call function
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                 #quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float32, # Ensure full precision.
                                                 device_map="cpu" # Explicitly set to CPU
                                                 )
    # Use the Tokenizer to encode the input text and store the result in a variable
    input_text = "What is the best recipe for Pepperoni pizza?"
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    # Generate text using the model and the tokenized input text
    # generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad(): # reduce memory usage during generation.
        generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text and print the result
    result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
    return result

