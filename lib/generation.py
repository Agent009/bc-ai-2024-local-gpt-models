from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from lib.utils import check_cuda
import torch

def run_gt():
    # check_cuda()
    default_input = "What is the best recipe for Pepperoni pizza?"
    user_input = input(f"Enter text for sentiment analysis (default: '{default_input}'): ")
    input_text = user_input if user_input else default_input
    output = generate_text_via_pipe(input_text)
    print(f"Generated text: " + output)

def generate_text_via_pipe(input_text):
    # If you have a GPU with at least 4GB of VRAM, you can use the device_map="auto" and
    # torch_dtype=torch.bfloat16 options to make the model run faster and use less memory
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds like an Italian chef",
        },
        {"role": "user", "content": input_text},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Configure the generation parameters for the model
    # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        # Start using the sampling method for generating text. Without this parameter, the model will use the greedy
        # search method, which always picks the most likely next word, which is not always the best choice.
        # Sampling is what gives the model the ability to generate different outputs each time it is called, based on
        # the randomness of the sampling method and the model's training data.
        # This parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling,
        # Top-K sampling and Top-p sampling, that are able to select the next tokens from the probability distribution
        # over the entire vocabulary with various strategy-specific adjustments
        do_sample=True,
        # Modulate the probabilities of sequential tokens. 0.5 is a good starting point for the task we're trying to
        # accomplish, but you can try different values to see how it affects the output
        temperature=0.5,
        # Broaden the range of tokens that the model can pick from. Consider the top 75 most likely tokens for the next
        # word, which will make the output a bit more diverse
        top_k=75,
        # Exclude less probable tokens from generation. Consider the top 90% most likely tokens for the next word,
        # which will probably make the output stay on topic and be more coherent
        top_p=0.9,
        # Avoid repeating the same 3 grams in the output, which will prevent repetitive and boring outputs.
        no_repeat_ngram_size=3,
    )
    return outputs[0]["generated_text"]


def generate_text(input_text):
    # We're going to use the AutoTokenizer class to automatically download the tokenizer suitable for the model we're using
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", padding_side="left")
    # Create a quantization configuration object to use with the model
    # bitsandbytes relies on CUDA, which is exclusively available for Nvidia GPUs.
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16
                                             # bnb_4bit_compute_dtype=torch.float32 # better CPU compatibility
                                             )
    # Create a new instance of the LLM model in a variable using a pre-trained model call function
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                 quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True,
                                                 # torch_dtype=torch.float32, # Ensure full precision.
                                                 device_map="auto"
                                                 # device_map="cpu" # Explicitly set to CPU
                                                 )
    # Use the Tokenizer to encode the input text and store the result in a variable
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

    # Generate text using the model and the tokenized input text
    # generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad(): # reduce memory usage during generation.
        generated_text = model.generate(**model_inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text and print the result
    result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
    return result

