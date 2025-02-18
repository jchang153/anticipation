import time, os, random

import midi2audio
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate, control_prefix
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi, midi_to_events_new, events_to_midi, compound_to_events, midi_to_compound_new
from anticipation.visuals import visualize # uses numpy < 2.0 which causes compatability errors with MLC
from anticipation.config import *
from anticipation.vocab import *
from anticipation.vocabs.tripletmidi import vocab
from anticipation.sample import _generate_live_chunk, _generate_live_chunk_no_cache
from anticipation.convert import make_events_safe
from anticipation.sample import nucleus, debugchat_forward

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    # Ignore on cluster. Needed for fluidsynth to work locally:
    import os
    # Add /opt/homebrew/bin/fluidsynth to PATH
    os.environ['PATH'] += ':/opt/homebrew/bin/'

from pathlib import Path
from mlc_llm.testing.debug_chat import DebugChat

### INITIALIZE MODEL

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

### CHORDER DEPENDENCIES

from chorder.chorder import Chord, Dechorder, chord_to_midi, play_chords
from miditoolkit import MidiFile
from copy import deepcopy
chord_program_num = vocab['chord_instrument'] - vocab['instrument_offset']

save_intermediate_midi_file = None

def extract_human_and_chords(midifile_path, human_program_num=None, return_non_human_events=False, remove_drums=True, relativize_time=True):
    chord_program_num = vocab['chord_instrument'] - vocab['instrument_offset']

    if return_non_human_events:
        assert human_program_num is not None, "Must provide human_program_num if return_non_human_events is True"

    if relativize_time:
        assert human_program_num is not None, "Must provide human_program_num if relativize_time is True; this is because the time offset is calculated based on the human part rn."

    if human_program_num is not None:
        # Extract human part
        events = midi_to_events_new(midifile_path, vocab)
        if remove_drums:
            events, _ = extract_instruments(events, [128])
        if relativize_time:
            min_time = -ops.min_time(events, seconds=False)
            events = ops.translate(events, min_time, seconds=False)
        non_human_events, human_events = extract_instruments(events, [human_program_num])
    else:
        human_events = None

    # Harmonize and assign chords to chord_program_num
    mf = MidiFile(midifile_path)
    if remove_drums:
        mf.instruments = [instr for instr in mf.instruments if instr.is_drum != True]
    mf_copy = deepcopy(mf) # chorder operations are done in-place
    for instr in mf_copy.instruments:
        if instr.program == human_program_num:
            mf_copy.instruments.remove(instr)
    mf_enchord = Dechorder.enchord(mf_copy)
    mf_chords = play_chords(mf_enchord) 
    mf_chords.instruments[0].program = chord_program_num
    mf.instruments = mf_chords.instruments # put back in original mf to preserve metadata
    global save_intermediate_midi_file
    save_intermediate_midi_file = mf
    mf.dump('tmp.mid')
    chord_events = compound_to_events(midi_to_compound_new('tmp.mid', vocab, debug=False)[0], vocab)
    _, chord_events = extract_instruments(chord_events, [chord_program_num])

    if relativize_time:
        chord_events = ops.translate(chord_events, min_time, seconds=False)

    if return_non_human_events:
        return (human_events, chord_events, non_human_events)

    return (human_events, chord_events)


### INITIALIZE PROMPT

filename = "b0ea637882ee7911da70d75f0b726992.mid"
human_instr = 0
original = "/Users/npb/Downloads/All The Things You Are.mid" #os.path.join("/Users/npb/Desktop/anticipation/lmd_full/b", filename)
original_events = midi_to_events_new(original)
# let's take out the drums
original_events, _ = extract_instruments(original_events, [128])
# remove silence in the beginning
original_events = ops.translate(original_events, -ops.min_time(original_events, seconds=False), seconds=False)

human_events, chord_events, agent_events = extract_human_and_chords(original, human_program_num=human_instr, return_non_human_events=True, relativize_time=True)

# ONLY FOR ALL THE THINGS YOU ARE!!
agent_events, _ = extract_instruments(agent_events, [35], as_controls=False)

def jitter(all_events, time_range, dur_range, only_instrs=None):
    events = []

    random.seed(42)

    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):

        is_control = dur > vocab['duration_offset'] + vocab['config']['max_duration'] and note > vocab['note_offset'] + vocab['config']['max_note']

        if only_instrs:
            # skip all instruments that are not chosen instrument
            instr = (note-vocab['note_offset'])//2**7
            if instr not in only_instrs: 
                events.extend([time, dur, note])
                continue
        else:
            # skip chords
            instr = (note-vocab['note_offset'])//2**7
            if instr in [vocab['chord_instrument']-vocab['instrument_offset']]:
                events.extend([time, dur, note])
                continue

        # randomly jitter note onsets, take care to make sure that there isn't overflow
        time_jitter = int(random.gauss(0.0, sigma=time_range/3.0))
        # commented out to let time token overflow for long songs:
        # new_time = min(max(time + time_jitter, vocab['time_offset'] + 1), vocab['time_offset'] + vocab['config']['max_time'] - 1) 
        if not is_control:
            new_time = max(time + time_jitter, vocab['time_offset'])
        else:
            new_time = max(time + time_jitter, vocab['atime_offset'])

        # randomly jitter note durations, take care to make sure that there isn't overflow
        dur_jitter = int(random.gauss(0.0, sigma=dur_range/3.0))
        if not is_control:
            new_dur = min(max(dur + dur_jitter, vocab['duration_offset']), vocab['duration_offset'] + vocab['config']['max_duration'] - 1)
        else:  
            new_dur = min(max(dur + dur_jitter, vocab['aduration_offset']), vocab['aduration_offset'] + vocab['config']['max_duration'] - 1)

        events.append([new_time, new_dur, note])

    events.sort(key=lambda x: x[0])
    events = [item for sublist in events for item in sublist]
    return events

def sort_tokens(tokens):
    sublists = [tokens[i:i+3] for i in range(0, len(tokens), 3)]
    sorted_sublists = sorted(sublists, key=lambda x: (x[0], x[1], x[2]))
    flattened_list = [token for sublist in sorted_sublists for token in sublist]
    return flattened_list

human_events = jitter(human_events, 4, 3)
agent_events = jitter(agent_events, 4, 3)

### SIMULATE HUMAN INPUT

clock_start = time.time()

simulation_start_time = 8 # NOTE: in the plugin, this function is triggered a second before simulation_start_time!
simulation_end_time = 21

GENERATION_INTERVAL = 2

use_MLC = False
use_file = False
impose_sorting = True

inputs = ops.clip(agent_events, 0, simulation_start_time, clip_duration=True, seconds=True)

for st in range(simulation_start_time, simulation_end_time+1, GENERATION_INTERVAL):

    accompaniment = [] # TODO: the way recursive inputs are handled may need to change for generation that spans multiple context windows

    start_time = st
    end_time = st + GENERATION_INTERVAL

    # Get all inputs that the plugin would see at start_time
    human_controls = ops.clip(human_events, 0, start_time-GENERATION_INTERVAL, clip_duration=True, seconds=True)

    # Dumb heuristic to deal with streaming not being handled correctly
    if len(inputs) + len(human_controls) + len(ops.clip(chord_events, 0, start_time, seconds=True, clip_duration=False)) > 768:
        force_z_cont = True
        chord_controls = ops.clip(chord_events, DELTA, ops.max_time(chord_events, seconds=True), seconds=True, clip_duration=False)
    else:
        force_z_cont = False
        chord_controls = chord_events

    # Get all agent events that the plugin would see at start_time

    instruments = sorted(list(ops.get_instruments(agent_events).keys()))
    human_instruments = [human_instr]
    masked_instrs=list(set(range(129)) - set(instruments))

    if impose_sorting:
        inputs = sort_tokens(inputs)
        human_controls = sort_tokens(human_controls)
        chord_controls = sort_tokens(chord_controls)
        
    torch.manual_seed(150)

    if use_file:
        accompaniment = _generate_live_chunk(
            model_mlc if use_MLC else model, 
            inputs=file_inputs, 
            chord_controls=file_chord_controls, 
            human_controls=file_human_controls, 
            start_time=start_time, 
            end_time=end_time, 
            instruments=instruments, 
            human_instruments=human_instruments, 
            temperature=1.0,
            top_p=.99, 
            masked_instrs=masked_instrs,
            debug=False,
            use_MLC=use_MLC, 
            force_z_cont=force_z_cont,
            save_input_ids_and_logits=True)
    else:

        os.makedirs(f'generate_plugin_sim/inputs_as_parts', exist_ok=True)

        with open(f'generate_plugin_sim/inputs_as_parts/{start_time}_input_events_nb.txt', 'w') as f:
            f.write(str(inputs))
        
        with open(f'generate_plugin_sim/inputs_as_parts/{start_time}_chord_controls_nb.txt', 'w') as f:
            f.write(str(chord_controls))

        with open(f'generate_plugin_sim/inputs_as_parts/{start_time}_human_controls_nb.txt', 'w') as f:
            f.write(str(human_controls))

        with open(f'generate_plugin_sim/inputs_as_parts/{start_time}_human_events_nb.txt', 'w') as f:
            f.write(str(human_events))

        accompaniment = _generate_live_chunk_no_cache(
            model_mlc if use_MLC else model, 
            inputs=inputs, 
            chord_controls=chord_controls, 
            human_controls=human_controls, 
            start_time=start_time, 
            end_time=end_time, 
            instruments=instruments, 
            human_instruments=human_instruments, 
            temperature=1.0,
            top_p=1.0, 
            masked_instrs=masked_instrs,
            debug=False,
            use_MLC=use_MLC, 
            force_z_cont=force_z_cont,
            save_input_ids_and_logits=True)
        
    # Recursive input: add accompaniment to inputs
    inputs = accompaniment

    clock_end = time.time()
    generation_time = clock_end - clock_start
    if generation_time > GENERATION_INTERVAL:
        print("Time to generate slower than real-time!")
    clock_start = clock_end

ops.print_tokens(inputs)

with open(f'generate_plugin_sim/{simulation_start_time}_{simulation_end_time}_inputs.txt', 'w') as f:
    f.write(str(inputs))

# inputs_midi = events_to_midi(make_events_safe(inputs), vocab)
inputs_midi = events_to_midi(inputs, vocab)
inputs_midi.save('generate_plugin_sim/inputs.mid')
