import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import time


from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
VLLM_AVAILABLE = True

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === vLLM-optimized generate function ===
def generate_vllm_tokens_only(vllm_model, tokenizer, input_ids, max_new_tokens):
    """
    vLLM generation that works with token inputs (for timing compatibility)
    """
    # Convert tokens back to text for vLLM
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=True
    )
    
    # Generate with vLLM
    outputs = vllm_model.generate([prompt_text], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    # Tokenize the full response
    full_text = prompt_text + generated_text
    full_tokens = tokenizer(full_text, return_tensors="pt")
    
    return full_tokens.input_ids

# === Traditional generate function (fallback) ===
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === vLLM MODEL LOADING ===
    model_name = "meta-llama/Llama-3.2-3B-Instruct"   
    
    if VLLM_AVAILABLE:
        print("🚀 Loading model with vLLM for maximum performance...")
        # Clean up any existing vLLM instances
        try:
            destroy_model_parallel()
        except:
            pass
        
        from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
        set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head']) #A16W4

        # Configure vLLM for T4 GPU
        vllm_model = LLM(
            model=model_name,
            dtype="float16",  # Use FP16 for speed
            gpu_memory_utilization=0.8,  # Conservative for T4
            max_model_len=2048,  # Reduced for T4 memory constraints
            tensor_parallel_size=1,  # Single GPU
            quantization=None,  # Can try "awq" or "gptq" if available
            trust_remote_code=True,
            seed=0,
            # Optimizations for T4
            enable_prefix_caching=True,
            disable_log_stats=True,
            calculate_kv_scales=True,
            kv_cache_dtype="fp8",  # Add this for KV cache optimization
        )
        # Traditional model will be loaded only for PPL evaluation
        model = None
        use_vllm = True
        print("✅ vLLM model loaded successfully")

    else:
        print("📦 Loading traditional transformers model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        vllm_model = None
        use_vllm = False
            
    
    #####################################
    
    if not use_vllm:
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🔧 Using {'vLLM' if use_vllm else 'Traditional'} inference engine")
    
    # === WARMUP ===
    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("🔥 Warming up...")
    
    if use_vllm:
        # vLLM warmup
        for i in tqdm(range(3), desc="vLLM Warm Up"):  # Reduced warmup for vLLM
            _ = generate_vllm_tokens_only(vllm_model, tokenizer, input_ids, 64)
    # else:
    #     # Traditional warmup
    #     model.prefill_forward = model.forward
    #     
    #     # Set up StaticCache for traditional model
    #     from transformers import StaticCache
    #     past_key_values = StaticCache(
    #         config=model.config, 
    #         max_batch_size=1, 
    #         max_cache_len=max_new_tokens + 16, 
    #         device=model.device, 
    #         dtype=torch.float16
    #     )
    #     
    #     for i in tqdm(range(5), desc="Traditional Warm Up"):
    #         generated = generate(model, input_ids, past_key_values, max_new_tokens)
    #         past_key_values.reset()
    
    # === PERFORMANCE TESTING ===
    if use_vllm:
        prompt = "How to learn a new language?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tputs = []
        time_record = []

        print(f"🚀 Running {'vLLM' if use_vllm else 'Traditional'} inference tests...")

        for i in tqdm(range(10), desc="Test Inference"):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if use_vllm:
                # === vLLM Generation ===
                generated = generate_vllm_tokens_only(vllm_model, tokenizer, input_ids, max_new_tokens)
            else:
                # === Traditional Generation ===
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # Deterministic
                    use_cache=True,
                )

            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)

            # Calculate actual generated tokens
            actual_generated = generated[0][input_ids.shape[1]:].shape[0]
            tput = actual_generated / (elapsed_ms / 1000)
            time_record.append(elapsed_ms / 1000)
            tputs.append(tput)

        # === RESULTS ===
        response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        sorted_tputs = np.sort(tputs)[2:-2]
        org_tput = np.mean(sorted_tputs)

        print(f'Prompt: {prompt}\nResponse: {response}\n')

        print(f'Time Record: {time_record}')
        print(f'Throughput Record: {tputs} toks/s\n')

        ### Your final throughput result ###
        print(f'Throughput: {org_tput} toks/s')


    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
    
    # Cleanup
    if use_vllm and vllm_model is not None:
        try:
            del vllm_model
            destroy_model_parallel()
        except:
            pass
    
    if not use_vllm and model is not None:
        del model
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()