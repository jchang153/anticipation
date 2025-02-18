"""
API functions for sampling from anticipatory infilling models.
"""
import math

import torch
import torch.nn.functional as F
import tvm
import numpy as np

from tqdm import tqdm

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import * # TODO: Deprecate this
from anticipation.vocabs.tripletmidi import vocab


def safe_logits(logits, idx, curtime=None, allowed_control_pn=None):
    if allowed_control_pn is None:
        # don't generate controls
        logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') 
        logits[vocab['instrument_offset']:] = -float('inf') 
    else:
        # don't generate (pitch,instr) tokens that do not correspond to allowed_control_pn
        instr = allowed_control_pn
        logits[ANOTE_OFFSET:(ANOTE_OFFSET+instr*MAX_PITCH)] = -float('inf')
        logits[(ANOTE_OFFSET+(instr+1)*MAX_PITCH):SPECIAL_OFFSET] = -float('inf')  
        logits[vocab['instrument_offset']:] = -float('inf')

        # only generate anti-anticipated atime tokens 
        assert curtime is not None
        logits[ATIME_OFFSET+curtime:ATIME_OFFSET+MAX_TIME] = -float('inf')     
        
    logits[SPECIAL_OFFSET:] = -float('inf') # don't generate special tokens

    # don't generate stuff in the wrong time slot
    if idx % 3 == 0:
        logits[vocab['duration_offset'] : vocab['duration_offset'] + vocab['config']['max_duration']] = -float('inf')
        logits[vocab['note_offset']     : vocab['note_offset']     + vocab['config']['max_note']]     = -float('inf')
    elif idx % 3 == 1:
        logits[vocab['time_offset']     : vocab['time_offset']     + vocab['config']['max_time']]     = -float('inf')
        logits[vocab['note_offset']     : vocab['note_offset']     + vocab['config']['max_note']]     = -float('inf')
    elif idx % 3 == 2:
        logits[vocab['time_offset']     : vocab['time_offset']     + vocab['config']['max_time']]     = -float('inf')
        logits[vocab['duration_offset'] : vocab['duration_offset'] + vocab['config']['max_duration']] = -float('inf')

    return logits


def nucleus(logits, top_p):
    # from HF implementation

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")               

    return logits


def future_logits(logits, curtime):
    """ don't sample events in the past """
    if curtime > 0:
        logits[TIME_OFFSET:TIME_OFFSET+curtime] = -float('inf')

    return logits


def instr_logits(logits, full_history):
    """ don't sample more than 16 instruments """
    instrs = ops.get_instruments(full_history)
    if len(instrs) < 16:
        return logits

    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits


def masked_instr_logits(logits, masked_instrs):
    """ supress the given instruments """
    for instr in masked_instrs:
        logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits

def control_prefix(instruments, human_instruments, task, vocab):
    task = vocab['task'][task]
    instr_offset = vocab['instrument_offset']
    human_instr_offset = vocab['human_instrument_offset']
    separator = vocab['separator']
    pad = vocab['pad']

    # get the list of instruments to condition on
    # by convention, let's provide the list sorted by instrument code
    instr_controls = sorted(instruments)
    instr_controls = [instr_offset + instr for instr in instruments]

    human_instr_controls = sorted(human_instruments)
    human_instr_controls = [human_instr_offset + instr for instr in human_instruments]

    instr_controls = instr_controls + human_instr_controls

    vocab_size = vocab['config']['size']
    assert max(instr_controls) < vocab_size

    # put task last, so the model knows it's time to generate events once it's seen the task token
    z_start = [separator] + instr_controls + [task]
    z_cont = instr_controls + [task]

    # pad the start controls out to an offset of 0 (mod 3)
    if len(z_start) % 3 > 0:
        z_start[1:1] = (3-len(z_start)%3)*[pad]

    # pad the continuation controls out to an offset of 1 (mod 3)
    if len(z_cont) % 3 > 0:
        z_cont[0:0] = (3-len(z_cont)%3)*[pad]
    z_cont = [pad] + z_cont

    return z_start, z_cont

def construct_prompt(instruments, human_instruments, task, tokens, cache, vocab, force_z_cont=False):
    pad = vocab['pad']

    # get control global control prefix for the beginning of a sequence and the continuation of a sequence
    task_string = 'autoregress' if task == [AUTOREGRESS] else 'anticipate'
    z_start, z_cont = control_prefix(instruments, human_instruments, task_string, vocab)

    history = tokens.copy()
    prefix = None

    if (len(tokens) + len(z_start) + 1) < 1024:
        lookback = 0
        if force_z_cont: # this is a hack to act like an continuation; see heuristic in live generation loop
            prefix = z_cont
        else:
            prefix = [pad] + z_start

    else:
        # if we hopped, flush the cache
        if (len(tokens) + len(z_start) + 1) == 1024 or ((len(tokens) + len(z_cont)) % 255 < 3):
            cache = None

        # compute quantized lookback for caching with z_cont
        lookback = max(len(tokens) + len(z_cont) - 768 - ((len(tokens) + len(z_cont)) % 255), 0)
        prefix = z_cont 

    history = history[lookback:] # Markov window
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    input_ids = torch.tensor(prefix + history)
    if cache:
        input_ids = input_ids[-1:]

    return input_ids, cache, offset

def add_token(model, task, tokens, instruments, human_instruments, top_p, temperature, current_time, masked_instrs, cache, allowed_control_pn=None, debug=False, use_MLC=False, force_z_cont=False, save_input_ids_and_logits=False):
    assert len(tokens) % 3 == 0

    new_token = []
    input_ids, cache, offset = construct_prompt(instruments, human_instruments, task, tokens, cache, vocab, force_z_cont=force_z_cont)
    with torch.no_grad():
        for i in range(3):
            if not use_MLC:
                if save_input_ids_and_logits:
                    import os
                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                    with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}_{i}.txt', 'w') as f:
                        f.write(str(input_ids.tolist()))
                input_ids = input_ids.unsqueeze(0).to(model.device)
                output = model(input_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[0,-1]
                if save_input_ids_and_logits:
                    import os
                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                    with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                        f.write(str(logits.tolist()))
            else:
                logits, cache = debugchat_forward(model, input_ids, cache)
                logits = torch.tensor(logits)[0,0,:]

            # og_logits = logits.clone()
            
            idx = len(tokens) + i
            logits = safe_logits(logits, idx, allowed_control_pn)

            # safel_logits = logits.clone()
            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)

            # instrl_logits = logits.clone()
            logits = masked_instr_logits(logits, masked_instrs)
            logits = nucleus(logits, top_p)
                
            probs = F.softmax(logits/temperature, dim=-1)
            # if use_MLC: # in torch 2.0.1, torch.multinomial has a bug on CPU where it samples zero prob events
            #     probs.to(str(model.device)[:-3]) 
            # else:
            #     probs.to(model.device)
            input_ids = torch.multinomial(probs, 1)
            new_token.append(int(input_ids))

            # if (i == 1 and int(input_ids) < vocab['duration_offset']) or (int(input_ids)) > CONTROL_OFFSET:
            #     print(i, '\n')
            #     torch.save(probs, 'probs_test.txt')
            #     print("Post nucleus control logits in range: ", CONTROL_OFFSET, SPECIAL_OFFSET, torch.min(logits[CONTROL_OFFSET:SPECIAL_OFFSET]).item(), torch.max(logits[CONTROL_OFFSET:SPECIAL_OFFSET]).item())
            #     print("Probs limits in range: ", CONTROL_OFFSET, SPECIAL_OFFSET, torch.min(probs[CONTROL_OFFSET:SPECIAL_OFFSET]).item(), torch.max(probs[CONTROL_OFFSET:SPECIAL_OFFSET]).item())
            #     print("Control range zero? ", (probs[27513:55025] == 0).any())
            #     print("Time range zero? ", (probs[vocab['time_offset']:vocab['time_offset']+vocab['config']['max_time']] == 0).any())
            #     print("Dur range nonzero? ", (probs[vocab['duration_offset']:vocab['duration_offset']+vocab['config']['max_duration']] != 0).any())
            #     print("PitchInstr range zero? ", (probs[vocab['note_offset']:vocab['note_offset']+vocab['config']['max_note']] == 0).any())
            #     print('input_ids', input_ids, ' cache==None ', cache==None)
            #     print('Probs limits: ', torch.min(probs).item(), torch.max(probs).item())
            #     print('\n')
            #     print("og_Logits range:", torch.min(og_logits).item(), torch.max(og_logits).item())
            #     print("safel_Logits range:", torch.min(safel_logits).item(), torch.max(safel_logits).item())
            #     print("instrl_Logits range:", torch.min(instrl_logits).item(), torch.max(instrl_logits).item(), i)
            #     print(current_time, offset, current_time - offset)
            #     print("Logits range:", torch.min(logits).item(), torch.max(logits).item())
            #     print(tokens)
                

    new_token[0] += offset # revert to full sequence timing
    if debug:
        print(f'  OFFSET = {offset}, TIME = {tokens[::3][-5:]}')

    return new_token, cache

def generate(model, start_time, end_time, inputs=None, chord_controls=None, human_controls=None, instruments=None, human_instruments=None, top_p=1.0, temperature=1.0, masked_instrs=[], debug=False, chord_delta=DELTA*TIME_RESOLUTION, human_delta=HUMAN_DELTA*TIME_RESOLUTION, return_controls=False, allowed_control_pn=None, use_MLC=False, save_input_ids_and_logits=False):
    
    if inputs is None:
        inputs = []

    if chord_controls is None:
        chord_controls = []

    if human_controls is None:
        human_controls = []

    if instruments is None:
        raise ValueError('Must provide list of instruments')

    if human_instruments is None:
        raise ValueError('Must provide list of human instruments s')

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, seconds=False, clip_duration=False), start_time)

    # treat events beyond start_time as controls
    future = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), seconds=False, clip_duration=False)
    if debug:
        print('Future')
        ops.print_tokens(future)

    # clip chord controls that preceed the sequence
    chord_controls = ops.clip(chord_controls, DELTA, ops.max_time(chord_controls, seconds=False), seconds=False, clip_duration=False)

    if debug:
        print('Chord Controls')
        ops.print_tokens(chord_controls)
        print('Human Controls')
        ops.print_tokens(human_controls)

    # task = [ANTICIPATE] if len(chord_controls) > 0 or len(future) > 0 or len(human_controls) > 0 else [AUTOREGRESS]
    task = [AUTOREGRESS] # always autoregress for now!
    if debug:
        print('AR Mode' if task[0] == AUTOREGRESS else 'AAR Mode')

    # interleave the chord_controls and human_controls with the events
    # note that we merge future with chord_controls, as they are both anticipated
    # tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))
    tokens, chord_controls, human_controls = ops.anticipate_and_anti_anticipate(prompt, ops.sort(chord_controls + [CONTROL_OFFSET+token for token in future]), human_controls, chord_delta=chord_delta, human_delta=human_delta)
    
    if debug:
        print('Prompt')
        ops.print_tokens(tokens)

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)
    
    with tqdm(range(end_time-start_time)) as progress:
        if chord_controls:
            atime, adur, anote = chord_controls[0:3]
            anticipated_tokens = chord_controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        if human_controls:
            aatime, aadur, aanote = human_controls[0:3]
            anti_anticipated_tokens = human_controls[3:]
            anti_anticipated_time = aatime - ATIME_OFFSET
        else:
            # nothing to anti-anticipate
            anti_anticipated_time = math.inf

        cache = None
        while True:
            while (current_time >= anticipated_time - chord_delta) or (current_time >= anti_anticipated_time - human_delta):
                if (anticipated_time - chord_delta <= anti_anticipated_time - human_delta):

                    # update the cache
                    input_ids, cache, offset = construct_prompt(instruments, human_instruments, task, tokens, cache, vocab)
                    for new_token in [atime-offset, adur, anote]:
                        with torch.no_grad():
                            # run the model as if we were going to use its prediction
                            if not use_MLC:
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}.txt', 'w') as f:
                                        f.write(str(input_ids.tolist()))
                                input_ids = input_ids.unsqueeze(0).to(model.device)
                                output = model(input_ids, past_key_values=cache, use_cache=True)
                                cache = output.past_key_values
                                logits = output.logits[0,-1]
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(logits.tolist()))
                            else:
                                _, cache = debugchat_forward(model, input_ids, cache)

                        tokens.append(new_token)
                        input_ids = torch.tensor([new_token])

                    if debug:
                        note = anote - ANOTE_OFFSET
                        instr = note//2**7
                        print('A', atime - ATIME_OFFSET, adur - ADUR_OFFSET, instr, note - (2**7)*instr)

                    if len(anticipated_tokens) > 0:
                        atime, adur, anote = anticipated_tokens[0:3]
                        anticipated_tokens = anticipated_tokens[3:]
                        anticipated_time = atime - ATIME_OFFSET
                    else:
                        # nothing more to anticipate
                        anticipated_time = math.inf
                else:
                    # update the cache
                    input_ids, cache, offset = construct_prompt(instruments, human_instruments, task, tokens, cache, vocab)
                    for i, new_token in enumerate([aatime-offset, aadur, aanote]):
                        with torch.no_grad():
                            # run the model as if we were going to use its prediction
                            if not use_MLC:
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(input_ids.tolist()))
                                input_ids = input_ids.unsqueeze(0).to(model.device)
                                output = model(input_ids, past_key_values=cache, use_cache=True)
                                cache = output.past_key_values
                                logits = output.logits[0,-1]
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(logits.tolist()))
                            else:
                                _, cache = debugchat_forward(model, input_ids, cache)
                        tokens.append(new_token)
                        input_ids = torch.tensor([new_token])

                    if debug:
                        note = aanote - ANOTE_OFFSET
                        instr = note//2**7
                        print('A', aatime - ATIME_OFFSET, aadur - ADUR_OFFSET, instr, note - (2**7)*instr)

                    if len(anti_anticipated_tokens) > 0:
                        aatime, aadur, aanote = anti_anticipated_tokens[0:3]
                        anti_anticipated_tokens = anti_anticipated_tokens[3:]
                        anti_anticipated_time = aatime - ATIME_OFFSET
                    else:
                        # nothing more to anti-anticipate
                        anti_anticipated_time = math.inf

            new_token, cache = add_token(model, task, tokens, instruments, human_instruments, top_p, temperature, max(start_time,current_time), masked_instrs, cache, allowed_control_pn, debug, use_MLC, force_z_cont=force_z_cont, save_input_ids_and_logits=save_input_ids_and_logits)
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                new_pitch = new_note - (2**7)*new_instr
                print('C', new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    events, controls = ops.split(tokens)
    if return_controls:
        return ops.unpad(events), controls
    else:
        return ops.sort(ops.unpad(events) + future)

def debugchat_forward(
    dc,
    input_tokens,
    kv_caches
):
    """
    Parameters
    ----------
    dc : DebugChat
        The DebugChat object that contains the model and tokenizer
        for generating the response.
        
    input_tokens : List[str]
        Either a prompt to the model if kv_caches is None, or the last token.

    kv_caches :
    """

    assert((len(input_tokens) == 1 and kv_caches is not None) or (kv_caches is None))

    if kv_caches is None:
        input_tokens = tvm.nd.array(np.array(input_tokens).astype("int32"), device=dc.device)
        embedding, input_len = dc._embed(input_tokens)
        logits, kv_caches = dc._prefill(embedding, input_len)
    else:
        last_token = input_tokens[-1]
        logits = dc._decode(last_token, kv_caches)
    
    return logits.numpy(), kv_caches

def _generate_live_chunk(
        model, 
        start_time, 
        end_time, 
        inputs=None, 
        chord_controls=None, 
        human_controls=None, 
        instruments=None, 
        human_instruments=None, 
        top_p=1.0, 
        temperature=1.0, 
        masked_instrs=[], 
        debug=False, 
        chord_delta=DELTA*TIME_RESOLUTION, 
        human_delta=HUMAN_DELTA*TIME_RESOLUTION, 
        use_MLC=True,
        force_z_cont=False,
        save_input_ids_and_logits=False
    ):

    if inputs is None:
        inputs = []

    if chord_controls is None:
        chord_controls = []

    if human_controls is None:
        human_controls = []

    if instruments is None:
        raise ValueError('Must provide list of instruments')

    if human_instruments is None:
        raise ValueError('Must provide list of human instruments s')

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    chord_delta = DELTA*TIME_RESOLUTION
    human_delta = HUMAN_DELTA*TIME_RESOLUTION

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, seconds=False, clip_duration=False), start_time)

    task = [AUTOREGRESS] # task is hardcoded to autoregress in live models

    # interleave the chord_controls and human_controls with the events
    # note that we merge future with chord_controls, as they are both anticipated
    # tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))
    tokens, chord_controls, human_controls = ops.anticipate_and_anti_anticipate(prompt, chord_controls, human_controls, chord_delta=chord_delta, human_delta=human_delta)

    # snap.append(construct_prompt(instruments, human_instruments, task, tokens, None, vocab, force_z_cont=force_z_cont)[0])

    current_time = ops.max_time(prompt, seconds=False)

    if len(tokens) > 1024:
        print(f"t = {current_time}, Outer loop: CONTEXT LENGTH REACHED")

    # Main generation loop
    with tqdm(range(end_time-start_time)) as progress:
        if chord_controls:
            atime, adur, anote = chord_controls[0:3]
            anticipated_tokens = chord_controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        if human_controls:
            aatime, aadur, aanote = human_controls[0:3]
            anti_anticipated_tokens = human_controls[3:]
            anti_anticipated_time = aatime - ATIME_OFFSET
        else:
            # nothing to anti-anticipate
            anti_anticipated_time = math.inf

        cache = None
        while True:
            while (current_time >= anticipated_time - chord_delta) or (current_time >= anti_anticipated_time - human_delta):
                if (anticipated_time - chord_delta <= anti_anticipated_time - human_delta):

                    # update the cache
                    input_ids, cache, offset = construct_prompt(instruments, human_instruments, task, tokens, cache, vocab, force_z_cont=force_z_cont)
                    for i, new_token in enumerate([atime-offset, adur, anote]):
                        with torch.no_grad():
                            # run the model as if we were going to use its prediction
                            if not use_MLC:
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(input_ids.tolist()))
                                input_ids = input_ids.unsqueeze(0).to(model.device)
                                output = model(input_ids, past_key_values=cache, use_cache=True)
                                cache = output.past_key_values
                                logits = output.logits[0, -1]
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(logits.tolist()))
                            else:
                                _, cache = debugchat_forward(model, input_ids, cache)

                        tokens.append(new_token)
                        input_ids = torch.tensor([new_token])

                    if len(anticipated_tokens) > 0:
                        atime, adur, anote = anticipated_tokens[0:3]
                        anticipated_tokens = anticipated_tokens[3:]
                        anticipated_time = atime - ATIME_OFFSET
                    else:
                        # nothing more to anticipate
                        anticipated_time = math.inf
                else:
                    # update the cache
                    input_ids, cache, offset = construct_prompt(instruments, human_instruments, task, tokens, cache, vocab, force_z_cont=force_z_cont)
                    for i, new_token in enumerate([aatime-offset, aadur, aanote]):
                        with torch.no_grad():
                            # run the model as if we were going to use its prediction
                            if not use_MLC:
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(input_ids.tolist()))
                                input_ids = input_ids.unsqueeze(0).to(model.device)
                                output = model(input_ids, past_key_values=cache, use_cache=True)
                                cache = output.past_key_values
                                logits = output.logits[0, -1]
                                if save_input_ids_and_logits:
                                    import os
                                    os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                                    with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                                        f.write(str(logits.tolist()))
                            else:
                                _, cache = debugchat_forward(model, input_ids, cache)
                        tokens.append(new_token)
                        input_ids = torch.tensor([new_token])

                    if len(anti_anticipated_tokens) > 0:
                        aatime, aadur, aanote = anti_anticipated_tokens[0:3]
                        anti_anticipated_tokens = anti_anticipated_tokens[3:]
                        anti_anticipated_time = aatime - ATIME_OFFSET
                    else:
                        # nothing more to anti-anticipate
                        anti_anticipated_time = math.inf

            new_token, cache = add_token(
                model, task, tokens, instruments, human_instruments, top_p, temperature,
                max(start_time, current_time), masked_instrs, cache, allowed_control_pn=None, debug=False,
                use_MLC=use_MLC, force_z_cont=force_z_cont, save_input_ids_and_logits=save_input_ids_and_logits
            )
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)
        
            if len(tokens) > 1024:
                print(f"t = {current_time}, Inner loop: CONTEXT LENGTH REACHED")

    new_events, controls = ops.split(tokens)

    new_events = ops.sort(ops.unpad(new_events))

    return new_events

# Helpers for non-caching generation

def construct_prompt_no_cache(instruments, human_instruments, task, tokens, vocab, force_z_cont=False):
    pad = vocab['pad']

    # get control global control prefix for the beginning of a sequence and the continuation of a sequence
    task_string = 'autoregress' if task == [AUTOREGRESS] else 'anticipate'
    z_start, z_cont = control_prefix(instruments, human_instruments, task_string, vocab)

    history = tokens.copy()
    prefix = None

    if (len(tokens) + len(z_start) + 1) < 1024:
        lookback = 0
        if force_z_cont: # this is a hack to act like an continuation; see heuristic in live generation loop
            prefix = z_cont
        else:
            prefix = [pad] + z_start
    else:
        # compute lookback to stay within context window
        lookback = max(len(tokens) - (1024 - len(z_cont)), 0)
        prefix = z_cont

    history = history[lookback:] # Markov window
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    input_ids = torch.tensor(prefix + history)
    return input_ids, offset

def add_token_no_cache(model, task, tokens, instruments, human_instruments, top_p, temperature, current_time, masked_instrs, allowed_control_pn=None, debug=False, use_MLC=False, force_z_cont=False, save_input_ids_and_logits=False):
    assert len(tokens) % 3 == 0

    # MLC always requires cache? not using here for now

    new_token = []
    input_ids, offset = construct_prompt_no_cache(instruments, human_instruments, task, tokens, vocab, force_z_cont=force_z_cont)
    with torch.no_grad():
        for i in range(3):
            if save_input_ids_and_logits:
                import os
                os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                with open(f'generate_plugin_sim/input_ids_and_logits/input_ids_{len(tokens)}_{i}.txt', 'w') as f:
                    f.write(str(input_ids.tolist()))
            input_ids = input_ids.unsqueeze(0).to(model.device)
            output = model(input_ids)
            logits = output.logits[0,-1]
            if save_input_ids_and_logits:
                import os
                os.makedirs('generate_plugin_sim/input_ids_and_logits', exist_ok=True)
                with open(f'generate_plugin_sim/input_ids_and_logits/logits_{len(tokens)}_{i}.txt', 'w') as f:
                    f.write(str(logits.tolist()))
            
            idx = len(tokens) + i
            logits = safe_logits(logits, idx, allowed_control_pn)

            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)

            logits = masked_instr_logits(logits, masked_instrs)
            logits = nucleus(logits, top_p)
                
            probs = F.softmax(logits/temperature, dim=-1)
            input_ids = torch.multinomial(probs, 1)
            new_token.append(int(input_ids))


    new_token[0] += offset # revert to full sequence timing
    if debug:
        print(f'  OFFSET = {offset}, TIME = {tokens[::3][-5:]}')

    return new_token

def _generate_live_chunk_no_cache(
        model, 
        start_time, 
        end_time, 
        inputs=None, 
        chord_controls=None, 
        human_controls=None, 
        instruments=None, 
        human_instruments=None, 
        top_p=1.0, 
        temperature=1.0, 
        masked_instrs=[], 
        debug=False, 
        chord_delta=DELTA*TIME_RESOLUTION, 
        human_delta=HUMAN_DELTA*TIME_RESOLUTION,
        force_z_cont=False,
        save_input_ids_and_logits=False,
        use_MLC=False
    ):

    if inputs is None:
        inputs = []

    if chord_controls is None:
        chord_controls = []

    if human_controls is None:
        human_controls = []

    if instruments is None:
        raise ValueError('Must provide list of instruments')

    if human_instruments is None:
        raise ValueError('Must provide list of human instruments s')

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    chord_delta = DELTA*TIME_RESOLUTION
    human_delta = HUMAN_DELTA*TIME_RESOLUTION

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, seconds=False, clip_duration=False), start_time)

    task = [AUTOREGRESS] # task is hardcoded to autoregress in live models

    # interleave the chord_controls and human_controls with the events
    tokens, chord_controls, human_controls = ops.anticipate_and_anti_anticipate(prompt, chord_controls, human_controls, chord_delta=chord_delta, human_delta=human_delta)

    current_time = ops.max_time(prompt, seconds=False)

    if len(tokens) > 1024:
        print(f"t = {current_time}, Outer loop: CONTEXT LENGTH REACHED")

    # Main generation loop
    with tqdm(range(end_time-start_time)) as progress:
        if chord_controls:
            atime, adur, anote = chord_controls[0:3]
            anticipated_tokens = chord_controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        if human_controls:
            aatime, aadur, aanote = human_controls[0:3]
            anti_anticipated_tokens = human_controls[3:]
            anti_anticipated_time = aatime - ATIME_OFFSET
        else:
            # nothing to anti-anticipate
            anti_anticipated_time = math.inf

        while True:
            while (current_time >= anticipated_time - chord_delta) or (current_time >= anti_anticipated_time - human_delta):
                if (anticipated_time - chord_delta <= anti_anticipated_time - human_delta):
                    # Add anticipated tokens
                    tokens.extend([atime, adur, anote])

                    if len(anticipated_tokens) > 0:
                        atime, adur, anote = anticipated_tokens[0:3]
                        anticipated_tokens = anticipated_tokens[3:]
                        anticipated_time = atime - ATIME_OFFSET
                    else:
                        # nothing more to anticipate
                        anticipated_time = math.inf
                else:
                    # Add anti-anticipated tokens
                    tokens.extend([aatime, aadur, aanote])

                    if len(anti_anticipated_tokens) > 0:
                        aatime, aadur, aanote = anti_anticipated_tokens[0:3]
                        anti_anticipated_tokens = anti_anticipated_tokens[3:]
                        anti_anticipated_time = aatime - ATIME_OFFSET
                    else:
                        # nothing more to anti-anticipate
                        anti_anticipated_time = math.inf

            new_token = add_token_no_cache(
                model, task, tokens, instruments, human_instruments, top_p, temperature,
                max(start_time, current_time), masked_instrs, allowed_control_pn=None, debug=False,
                force_z_cont=force_z_cont, save_input_ids_and_logits=save_input_ids_and_logits
            )
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)
        
            if len(tokens) > 1024:
                print(f"t = {current_time}, Inner loop: CONTEXT LENGTH REACHED")

    new_events, controls = ops.split(tokens)
    new_events = ops.sort(ops.unpad(new_events))

    return new_events





def generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, debug=False, delta=DELTA*TIME_RESOLUTION):
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time)

    # treat events beyond start_time as controls
    future = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), clip_duration=False, seconds=False)
    if debug:
        print('Future')
        ops.print_tokens(future)

    # clip controls that preceed the sequence
    controls = ops.clip(controls, DELTA, ops.max_time(controls, seconds=False), clip_duration=False, seconds=False)

    if debug:
        print('Controls')
        ops.print_tokens(controls)

    z = [ANTICIPATE] if len(controls) > 0 or len(future) > 0 else [AUTOREGRESS]
    if debug:
        print('AR Mode' if z[0] == AUTOREGRESS else 'AAR Mode')

    # interleave the controls with the events
    tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))

    if debug:
        print('Prompt')
        ops.print_tokens(tokens)

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)

    with tqdm(range(end_time-start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            anticipated_tokens = controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        while True:
            while current_time >= anticipated_time - delta:
                tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - ANOTE_OFFSET
                    instr = note//2**7
                    print('A', atime - ATIME_OFFSET, adur - ADUR_OFFSET, instr, note - (2**7)*instr)

                if len(anticipated_tokens) > 0:
                    atime, adur, anote = anticipated_tokens[0:3]
                    anticipated_tokens = anticipated_tokens[3:]
                    anticipated_time = atime - ATIME_OFFSET
                else:
                    # nothing more to anticipate
                    anticipated_time = math.inf

            new_token = add_token(model, z, tokens, top_p, max(start_time,current_time))
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                new_pitch = new_note - (2**7)*new_instr
                print('C', new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    events, _ = ops.split(tokens)
    return ops.sort(ops.unpad(events) + future)