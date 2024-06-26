import os, math, traceback
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np

from anticipation import ops
from anticipation.tokenize import maybe_tokenize
from anticipation.config import DELTA, HUMAN_DELTA, TIME_RESOLUTION

def extract_instruments(all_events, instruments, vocab):
    events = []
    controls = []

    control_offset = vocab['control_offset']
    note_offset = vocab['note_offset']
    separator = vocab['separator']
    rest = vocab['rest']

    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < control_offset         # shouldn't be in the sequence yet
        assert note not in [separator, rest] # these shouldn't either

        instr = (note-note_offset)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([control_offset+time, control_offset+dur, control_offset+note])
        else:
            events.extend([time, dur, note])

    return events, controls

def prepare_triplet_midi(midifile, vocab):
    with open(midifile, 'r') as f:
        cmp = [int(token) for token in f.read().split()]
        events, truncations, status = maybe_tokenize(cmp, vocab)

    if status > 0:
        raise ValueError(f'Bad midi sequence (status {status})')

    return events

def control_prefix(instruments, human_instruments, task, vocab):

    task = vocab['task'][task]
    instr_offset = vocab['instrument_offset']
    separator = vocab['separator']
    pad = vocab['pad']

    # get the list of instruments to condition on
    # by convention, let's provide the list sorted by instrument code
    instr_controls = sorted(instruments)
    instr_controls = [instr_offset + instr for instr in instruments]

    if human_instruments is not None:
        human_instr_offset = vocab['human_instrument_offset']
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

def extract_spans(all_events, rate, vocab):
    events = []
    controls = []
    span = True
    next_span = end_span = vocab['time_offset']+0
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [vocab['separator'], vocab['rest']]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and time >= end_span:
            span = False
            next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

        # anticipate a 3-second span
        if (not span) and time >= next_span:
            span = True
            end_span = time + DELTA*TIME_RESOLUTION

        if span:
            # mark this event as a control
            controls.extend([vocab['control_offset']+time, vocab['control_offset']+dur, vocab['control_offset']+note])
        else:
            events.extend([time, dur, note])

    return events, controls

ANTICIPATION_RATES = 10
def extract_random(all_events, rate, vocab):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [vocab['separator'], vocab['rest']]) # shouldn't be in the sequence yet

        if np.random.random() < rate/float(ANTICIPATION_RATES):
            # mark this event as a control
            controls.extend([vocab['control_offset']+time, vocab['control_offset']+dur, vocab['control_offset']+note])
        else:
            events.extend([time, dur, note])

    return events, controls

def pack_tokens(sequences, output, idx, vocab, prepare, prefix, seqlen, live=False, piano_human_part=True):
    vocab_size = vocab['config']['size']
    pad = vocab['pad']
    files = bad_files = seqcount = 0
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for sequence in tqdm(sequences, desc=f'#{idx}', position=idx+1, leave=True):
            if len(concatenated_tokens) == 0:
                z = [pad]

            try:
                events = prepare(sequence)
                files += 1
            except Exception as e:
                #print(e)
                #print(traceback.format_exc())
                bad_files += 1
                continue

            # record the original end time before extracting control tokens
            end_time = ops.max_time(events, seconds=False)          
            instruments = sorted(list(ops.get_instruments(events).keys()))

            if live:
                chords_program_num = vocab['chord_instrument'] - vocab['instrument_offset']

                # extract the chord sequence to anticipate
                chord_controls = None
                if chords_program_num in instruments:
                    if len(instruments) < 3:
                        continue
                    events, chord_controls = extract_instruments(events, [chords_program_num], vocab)
                    instruments.remove(chords_program_num)
                else:
                    if len(instruments) < 2:
                        continue

                # extract piano or randomly selected "human" sequence to anti-anticipate
                if piano_human_part:
                    human = [0]
                else:
                    human = np.random.choice(instruments, 1, replace=False)

                instruments.remove(human[0])
                events, human_controls = extract_instruments(events, human, vocab)

                # get the global control tokens for this sequence
                # do this before padding because some ops don't handle REST properly
                z_start, z_cont = prefix(instruments, human)

                # add rest tokens to events after extracting control tokens
                # (see Section 3.2 of the paper for why we do this)
                events = ops.pad(events, end_time)

                # interleave control tokens
                tokens, chord_controls, human_controls = ops.anticipate_and_anti_anticipate(events, chord_controls, human_controls, chord_delta=DELTA*TIME_RESOLUTION, human_delta=HUMAN_DELTA*TIME_RESOLUTION)

                # write out full contexts to file
                concatenated_tokens.extend(z_start + tokens)
                while len(concatenated_tokens) >= seqlen-len(z):
                    seq = concatenated_tokens[0:seqlen-len(z)]
                    concatenated_tokens = concatenated_tokens[len(seq):]

                    # relativize time to the context 
                    seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                    assert ops.min_time(seq, seconds=False) == 0

                    # if notes in the chunk exceed vocab max time, skip it
                    if ops.max_time(seq, seconds=False) >= vocab['config']['max_time']:
                        continue

                    seq = z + seq

                    assert max(seq) < vocab_size
                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                    z = z_cont # update the global control prompt (if it changed)
                    seqcount += 1
            else:

                all_events = events.copy()

                # different random augmentations
                augment_factor = 30
                for k in range(augment_factor): # using default augment_factor from original paper
                    if k % 10 == 0:
                        # no augmentation
                        events = all_events.copy()
                        controls = []
                    elif k % 10 == 1:
                        # span augmentation
                        lmbda = .05
                        events, controls = extract_spans(all_events, lmbda, vocab)
                    elif k % 10 < 6:
                        # random augmentation
                        r = np.random.randint(1,ANTICIPATION_RATES)
                        events, controls = extract_random(all_events, r, vocab)
                    else:
                        if len(instruments) > 1:
                            # instrument augmentation: at least one, but not all instruments
                            u = 1+np.random.randint(len(instruments)-1)
                            subset = np.random.choice(instruments, u, replace=False)
                            events, controls = extract_instruments(all_events, subset, vocab)
                        else:
                            # no augmentation
                            events = all_events.copy()
                            controls = []

                    z_start, z_cont = prefix(instruments, None)
                    events = ops.pad(events, end_time)
    
                    tokens, controls = ops.anticipate(events, controls, DELTA*TIME_RESOLUTION)
                    assert len(controls) == 0 # should have consumed all controls (because of padding)

                    # write out full contexts to file
                    concatenated_tokens.extend(z_start + tokens)
                    while len(concatenated_tokens) >= seqlen-len(z):
                        seq = concatenated_tokens[0:seqlen-len(z)]
                        concatenated_tokens = concatenated_tokens[len(seq):]

                        # relativize time to the context 
                        seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                        assert ops.min_time(seq, seconds=False) == 0

                        # if notes in the chunk exceed vocab max time, skip it
                        if ops.max_time(seq, seconds=False) >= vocab['config']['max_time']:
                            continue

                        seq = z + seq

                        assert max(seq) < vocab_size
                        outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                        z = z_cont # update the global control prompt (if it changed)
                        seqcount += 1

    return (files, bad_files, seqcount)


def preprocess_midi(midifiles, output, seqlen, task, vocab, idx, live=False):
    prefix = lambda instruments, human_instruments: control_prefix(instruments, human_instruments, task, vocab)
    prepare = lambda mid: prepare_triplet_midi(mid, vocab)

    return pack_tokens(midifiles, output, idx, vocab, prepare, prefix, seqlen=seqlen, live=live)


preproc_func = {
    'autoregress' : preprocess_midi,
}

def main(args):
    print('Tokenizing a dataset at:', args.datadir)

    if args.vocab == 'triplet-midi':
        from anticipation.vocabs.tripletmidi import vocab
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')
    
    print('Tokenization parameters:')
    print(f"  vocab = {args.vocab}")
    print(f"  task = {args.task}")
    print(f"  context = {args.context}")
    print(f"  anticipation interval = {vocab['config']['anticipation']} seconds")
    print(f"  anti-anticipation interval = {vocab['config']['anti-anticipation']} seconds")
    print(f"  skew = {vocab['config']['skew']}")

    files = glob(os.path.join(args.datadir, '**/*.compound.txt'), recursive=True)

    n = len(files) // args.workers
    shards = [files[i*n:(i+1)*n] for i in range(args.workers)] # dropping a few tracks (< args.workers)
    outfiles = os.path.join(args.outdir, os.path.basename(args.datadir) + '.{t}.shard-{s:03}.txt')
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(t=args.task, s=s) for s in range(len(shards))]
    context = args.workers*[args.context]
    task = args.workers*[args.task]
    vocab = args.workers*[vocab]

    print('Processing...')
    if args.debug:
        results = preproc_func[args.task](shards[0], outputs[0], args.context, args.task, vocab[0], 0, args.live)
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            results = pool.starmap(preproc_func[args.task], zip(shards, outputs, context, task, vocab, range(args.workers)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a dataset')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('task', help='task for which we are preparing sequences')
    parser.add_argument('context', type=int, default=1024, help='context length for packing training sequences')
    parser.add_argument('-v', '--vocab', default='triplet-midi', help='name of vocabulary to use for tokenization')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')
    parser.add_argument('--live', action='store_true', help='tokenize for live mode (chords and human part)')

    main(parser.parse_args())
