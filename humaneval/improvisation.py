import os, csv, time

from argparse import ArgumentParser

import numpy as np

from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate, generate_live_eval
from anticipation.tokenize import extract_instruments
from anticipation.convert import midi_to_events, events_to_midi, compound_to_events, midi_to_compound
from anticipation.config import DELTA, HUMAN_DELTA, TIME_RESOLUTION
from anticipation.vocabs.tripletmidi import vocab

from chorder.chorder import Chord, Dechorder, chord_to_midi, play_chords
from miditoolkit import MidiFile
from copy import deepcopy

np.random.seed(0)

def extract_human_and_chords(midifile_path, human_program_num=None, return_non_human_events=False):
    chord_program_num = vocab['chord_instrument'] - vocab['instrument_offset']

    if human_program_num:
        # Extract human part
        events = midi_to_events(midifile_path, vocab)
        non_human_events, human_events = extract_instruments(events, [human_program_num])
    else:
        human_events = None

    # Harmonize and assign chords to chord_program_num
    mf = MidiFile(midifile_path)
    mf_copy = deepcopy(mf) # chorder operations are done in-place
    for instr in mf_copy.instruments:
        if instr.program == human_program_num:
            mf_copy.instruments.remove(instr)
    mf_enchord = Dechorder.enchord(mf_copy)
    mf_chords = play_chords(mf_enchord) 
    mf_chords.instruments[0].program = chord_program_num
    mf.instruments = mf_chords.instruments # put back in original mf to preserve metadata
    mf.dump('tmp.mid')
    chord_events = compound_to_events(midi_to_compound('tmp.mid', vocab, debug=False), vocab)
    _, chord_events = extract_instruments(chord_events, [chord_program_num])

    if return_non_human_events:
        return (human_events, chord_events, non_human_events)

    return (human_events, chord_events)

def main(args):

    print(f'Using model checkpoint: {args.model}')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    print(f'Loaded model ({time.time()-t0} seconds)')

    print(f'Writing outputs to {args.dir}/pairs')
    try:
        os.makedirs(f'{args.dir}/pairs')
    except FileExistsError:
        pass

    print(f'Improvising with tracks in index : {args.dir}/index.csv')
    with open(f'{args.dir}/index.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:

            try: 
                original = os.path.join(args.midis, row[header.index('original')])
                clipped_midi = row[header.index('clip')]
                melody = int(row[header.index('melody')])
                idx = int(row[header.index('idx')])

                if idx < 37:
                    continue
                    
                start_time = float(row[header.index('start_time')])
                end_time = float(row[header.index('end_time')])
    
                human_events, chord_events, non_human_events = extract_human_and_chords(os.path.join(args.dir, clipped_midi), human_program_num=melody, return_non_human_events=True)
    
                requested_instruments = sorted(list(ops.get_instruments(non_human_events).keys()))
                human_instruments = [melody]
                prompt_length = args.prompt_length
    
                events = midi_to_events(os.path.join(args.dir, clipped_midi))
                min_notes = float("inf")
                min_instr = list(ops.get_instruments(events).keys())[0]
    
                for inst in ops.get_instruments(events):
                    num = ops.get_instruments(events)[inst]
                    if num < min_notes:
                        min_notes = num
                        min_instr = inst   
    
                human_events1, chord_events1, non_human_events1 = extract_human_and_chords(os.path.join(args.dir, clipped_midi), human_program_num=min_instr, return_non_human_events=True)
                requested_instruments1 = sorted(list(ops.get_instruments(non_human_events1).keys()))
    
                # ========================================================================================
    
                for j in range(args.multiplicity):
                    t0 = time.time()
    
                    try:
    
                        # Generate alt melody 1
                          
                        # human_controls = ops.clip(human_events,     0, start_time+prompt_length, seconds=True)
                        # inputs         = ops.clip(non_human_events, 0, start_time+prompt_length, seconds=True)
                        # chord_controls = ops.clip(chord_events,     0, end_time,                 seconds=True)
                        
                        # events, controls = generate_live_eval(
                        #     model, 
                        #     start_time, 
                        #     end_time, 
                        #     prompt_length, 
                        #     inputs=inputs, 
                        #     chord_controls=chord_controls, 
                        #     human_controls=human_controls, 
                        #     instruments=requested_instruments, 
                        #     human_instruments=human_instruments, 
                        #     top_p=1.0, 
                        #     temperature=1.0, 
                        #     debug=False, 
                        #     chord_delta=DELTA*TIME_RESOLUTION, 
                        #     human_delta=HUMAN_DELTA*TIME_RESOLUTION, 
                        #     return_controls=True, 
                        #     allowed_control_pn=human_instruments[0])
                        
                        # _, alt_melody_1 = extract_instruments([tok - vocab['control_offset'] for tok in controls], human_instruments, as_controls=False)
        
                        human_controls = ops.clip(human_events1,     0, start_time+prompt_length, seconds=True)
                        inputs         = ops.clip(non_human_events1, 0, start_time+prompt_length, seconds=True)
                        chord_controls = ops.clip(chord_events1,     0, end_time,                 seconds=True) 
                        
                        events = generate(
                            model, 
                            inputs=inputs, 
                            chord_controls=chord_controls, 
                            human_controls=human_controls, 
                            start_time=start_time+prompt_length, 
                            end_time=end_time, 
                            instruments=requested_instruments1, 
                            human_instruments=[min_instr], 
                            top_p=.99, 
                            masked_instrs=list(set(range(129)) - set(requested_instruments1)),
                            allowed_control_pn=None)
        
                        _, alt_melody_1 = extract_instruments(events, human_instruments, as_controls=False)
                        
                        mid = events_to_midi(ops.translate(ops.clip(alt_melody_1, start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-alt-melody-{j}.mid')
        
                        # Generate accompaniment to original human part
        
                        human_controls = ops.clip(human_events,     0, end_time,                 seconds=True)
                        inputs         = ops.clip(non_human_events, 0, start_time+prompt_length, seconds=True)
                        chord_controls = ops.clip(chord_events,     0, end_time,                 seconds=True) 
        
                        accompaniment = generate(
                            model, 
                            inputs=inputs, 
                            chord_controls=chord_controls, 
                            human_controls=human_controls, 
                            start_time=start_time+prompt_length, 
                            end_time=end_time, 
                            instruments=requested_instruments, 
                            human_instruments=human_instruments, 
                            top_p=.99, 
                            masked_instrs=list(set(range(129)) - set(requested_instruments)),
                            allowed_control_pn=None)
        
                        mid = events_to_midi(ops.translate(ops.clip(accompaniment, start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-accompaniment-{j}.mid')
        
                        # Save combined melody+accompaniment and ground truth
        
                        mid = events_to_midi(ops.translate(ops.clip([tok - vocab['control_offset'] for tok in human_events], start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-melody-{j}.mid')
        
                        mid = events_to_midi(ops.translate(ops.clip(ops.sort(accompaniment + [tok - vocab['control_offset'] for tok in human_events]), start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-combined-{j}.mid')
        
                        mid = events_to_midi(ops.translate(ops.clip(ops.sort(accompaniment + alt_melody_1), start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-alt-combined-{j}.mid')
        
                        mid = events_to_midi(ops.translate(ops.clip(midi_to_events(os.path.join(args.dir, clipped_midi), vocab), start_time, end_time), -start_time, seconds=True), vocab)
                        mid.save(f'{args.dir}/pairs/{idx}-gt-{j}.mid')
                        
                        print(f'Generated clips for: {idx}. Sampling time: {time.time()-t0} seconds')
    
                    except Exception as e:
                        # Handle the exception
                        print(f"An error occurred: {str(e)}")
                        # Continue with the next iteration of the loop

            except Exception as e:
                print(f"An error occurred: {str(e)}")
        


if __name__ == '__main__':
    parser = ArgumentParser(description='generate infilling completions for live model')
    parser.add_argument('dir', help='directory containing an index of MIDI files')
    parser.add_argument('--model', type=str, default='',
            help='directory containing an anticipatory model checkpoint')
    parser.add_argument('-m', '--multiplicity', type=int, default=1,
            help='number of generations per clip')
    parser.add_argument('-p', '--prompt_length', type=int, default=10,
            help='length of the prompt (in seconds)')
    parser.add_argument('-l', '--clip_length', type=int, default=30,
            help='length of the full clip (in seconds)')
    parser.add_argument('-d', '--midis', type=str, default='',
            help='directory containing the reference MIDI files (for retrieval)')
    main(parser.parse_args())
