
# the /inspection_outputs folder has the logits of the entire sequence in this run
import os
import time
from scipy.stats import entropy
import torch
import torch.nn.functional as F

import multiprocessing as mp
import torch.nn.functional as F

import glob
import ast

from pathlib import Path
from mlc_llm.testing.debug_chat import DebugChat
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from anticipation.sample import nucleus, debugchat_forward


# Models

# HF models
# AMT_MED = '/juice4/scr4/nlp/music/lakh-checkpoints/futile-think-tank-272/step-800000/hf'
# INST_MODEL = '/juice4/scr4/nlp/music/prelim-checkpoints/triplet-live/step-98844/hf/' # from Feb
INSTR_MED_BASELINE_HF = '/juice4/scr4/nlp/music/prelim-checkpoints/instr-finetune-30/0ha1twnc/step-2000/hf'
INSTR_MED_BASELINE_AR_HF = '/juice4/scr4/nlp/music/prelim-checkpoints/instr-finetune-autoreg/7cxypt7a/step-2000/hf'
# LIVE = '/juice4/scr4/nlp/music/prelim-checkpoints/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/hf'

# MLC models
INSTR_MED_BASELINE_AR_MLC = '/juice4/scr4/nlp/music/prelim-checkpoints/instr-finetune-autoreg/7cxypt7a/step-2000/mlc'
INSTR_MED_BASELINE_AR_MLC_LIB = '/juice4/scr4/nlp/music/prelim-checkpoints/instr-finetune-autoreg/7cxypt7a/step-2000/mlc/instr-finetune-autoreg-med.so'

# LIVE_MLC = '/juice4/scr4/nlp/music/prelim-checkpoints/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/mlc/'
# LIVE_MLC_LIB = '/juice4/scr4/nlp/music/prelim-checkpoints/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/mlc/mlc_cuda.so'

# Local:
LIVE = '/Users/npb/Desktop/anticipation/anticipation/mlc_music_models/models/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/hf'
LIVE_MLC = '/Users/npb/Desktop/anticipation/anticipation/mlc_music_models/models/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/mlc'
LIVE_MLC_LIB = '/Users/npb/Desktop/anticipation/anticipation/mlc_music_models/models/live-finetune-piano-aug-0604-med/1eaqb2uc/step-2000/mlc/q0f16-metal.so'

# load an anticipatory music transformer
if not torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(LIVE)
else:
    model = AutoModelForCausalLM.from_pretrained(LIVE).cuda()

# load an anticipatory music transformer with MLC
class DummyDebugInstrument:
    def __init__(self, debug_out: Path):
        self.debug_out = debug_out
        pass

    def reset(self, debug_out: Path):
        pass

    def __call__(self, func, name, before_run, ret_val, *args):
        pass
        
model_mlc = DebugChat(
    model=LIVE_MLC,
    debug_dir=Path("./debug-anticipation"),
    model_lib=LIVE_MLC_LIB,
    debug_instrument=DummyDebugInstrument(Path("./debug-anticipation"))
)


# Per-sequence perplexity plot


def process_window(begin_loc, end_loc, prev_end_loc, prompt, use_MLC, model, model_mlc):
    print(f"\nWindow: begin_loc={begin_loc}, end_loc={end_loc}, trg_len={end_loc - prev_end_loc}")
    
    # Add batch dimension
    input_ids = torch.tensor(prompt[begin_loc:end_loc]).unsqueeze(0)  # Shape: [1, seq_len]
    target_ids = input_ids.clone()
    target_ids[0, :-end_loc + prev_end_loc] = -100  # Apply to first batch dimension
    print(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")

    with torch.no_grad():
        if not use_MLC:
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            print(f"HF model loss: {neg_log_likelihood:.4f}")
        else:
            # For MLC model, manually compute loss
            logits, _ = debugchat_forward(model_mlc, input_ids[0], None)  # Remove batch dim for MLC
            logits = torch.tensor(logits).unsqueeze(0)  # Add batch dim back: [1, seq_len, vocab]
            print(f"MLC logits shape: {logits.shape}")

            shift_logits = logits[:, :-1].contiguous()  # [1, seq_len-1, vocab]
            shift_labels = target_ids[:, 1:].contiguous()  # [1, seq_len-1]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                       shift_labels.view(-1))
            print(f"MLC model loss: {neg_log_likelihood:.4f}")

    num_valid_tokens = (target_ids != -100).sum().item()
    batch_size = target_ids.size(0)
    num_loss_tokens = num_valid_tokens - batch_size
    print(f"Valid tokens: {num_valid_tokens}, Batch size: {batch_size}, Loss tokens: {num_loss_tokens}")

    return neg_log_likelihood, num_loss_tokens

def process_prompt(prompt, use_MLC, model, model_mlc, max_length=1024, stride=512):
    print(f"\nProcessing prompt of length {len(prompt)}")
    seq_len = len(prompt)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    # Create pool of worker processes
    with mp.Pool() as pool:
        # Prepare arguments for each window
        tasks = []
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            args = (begin_loc, end_loc, prev_end_loc, prompt, use_MLC, model, model_mlc)
            tasks.append(pool.apply_async(process_window, args))
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            
        # Collect results
        for task in tasks:
            neg_log_likelihood, num_loss_tokens = task.get()
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens
            print(f"Running totals - NLL sum: {nll_sum:.4f}, Total tokens: {n_tokens}")

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    print(f"\nFinal metrics - Avg NLL: {avg_nll:.4f}, Perplexity: {ppl:.4f}")
    return ppl.item()

sequence_perplexities = []
use_MLC = False

def main():

    # Input files 

    input_ids_files = sorted(glob.glob('inspection_outputs/input_ids_*_*.txt'), key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))
    logits_files = sorted(glob.glob('inspection_outputs/logits_*_*.txt'), key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))

    def process_input_file(file):
        with open(file, 'r') as f:
            return ast.literal_eval(f.read())

    def process_logits_file(file):
        with open(file, 'r') as f:
            return torch.tensor(ast.literal_eval(f.read()))

    # Process input files sequentially
    tokens_list = []
    for file in tqdm(input_ids_files, desc="Processing input files"):
        tokens_list.append(process_input_file(file))

    # Process tokens into prompts
    prompts = []
    current_prompt = []
    for tokens in tqdm(tokens_list, desc="Processing tokens into prompts"):
        if isinstance(tokens, list) and len(tokens) > 1:
            current_prompt = tokens
        else:
            current_prompt = current_prompt + tokens
            prompts.append(current_prompt.copy())

    print(f"Found {len(prompts)} prompts")

    # Parallel processing of logits files
    # all_logits = [process_logits_file(file) for file in logits_files]

    for prompt in prompts:
        ppl = process_prompt(prompt, use_MLC, model, model_mlc)
        sequence_perplexities.append(ppl)

    # Save perplexity results
    os.makedirs('prompt/analysis', exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f'prompt/analysis/perplexities_{timestamp}.txt'
    
    with open(save_path, 'w') as f:
        for i, ppl in enumerate(sequence_perplexities):
            f.write(f"Sequence {i}: {ppl}\n")
    
    print(f"\nPerplexity results saved to {save_path}")

if __name__ == '__main__':
    main()
