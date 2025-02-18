"""
Utilities for converting to and from Midi data and encoded/tokenized data.
"""

from collections import defaultdict

import mido
import miditoolkit

from anticipation.config import *
# from anticipation.vocab import *
from anticipation.ops import unpad
from anticipation.vocabs.tripletmidi import vocab

import sys
# sys.path.append('/Users/npb/Desktop/anticipation/anticipation/chorder/')
# from chorder import Chord, Dechorder, chord_to_midi, play_chords
# Instead of the above, make sure tld of chorder has __init__.py:
from chorder.chorder import Chord, Dechorder, chord_to_midi, play_chords
from copy import deepcopy

from math import floor


def midi_to_interarrival(midifile, debug=False, stats=False):
    midi = mido.MidiFile(midifile)

    tokens = []
    dt = 0

    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat
    truncations = 0
    for message in midi:
        dt += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ['note_on', 'note_off']:
            delta_ticks = min(round(TIME_RESOLUTION*dt), MAX_INTERARRIVAL-1)
            if delta_ticks != round(TIME_RESOLUTION*dt):
                truncations += 1

            if delta_ticks > 0: # if time elapsed since last token
                tokens.append(MIDI_TIME_OFFSET + delta_ticks) # add a time step event

            # special case: channel 9 is drums!
            inst = 128 if message.channel == 9 else instruments[message.channel]
            offset = MIDI_START_OFFSET if message.type == 'note_on' and message.velocity > 0 else MIDI_END_OFFSET
            tokens.append(offset + (2**7)*inst + message.note)
            dt = 0
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    if stats:
        return tokens, truncations

    return tokens


def interarrival_to_midi(tokens, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2 # 2 beats/second at quarter=120

    track_idx = {} # maps instrument to (track number, current time)
    time_in_ticks = 0
    num_tracks = 0
    for token in tokens:
        if token == MIDI_SEPARATOR:
            continue

        if token < MIDI_START_OFFSET:
            time_in_ticks += token - MIDI_TIME_OFFSET
        elif token < MIDI_END_OFFSET:
            token -= MIDI_START_OFFSET
            instrument = token // 2**7
            pitch = token - (2**7)*instrument

            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                idx = num_tracks
                previous_time = 0
                track = mido.MidiTrack()
                mid.tracks.append(track)
                if instrument == 128: # drums always go on channel 9
                    idx = 9
                    message = mido.Message('program_change', channel=idx, program=0)
                else:
                    message = mido.Message('program_change', channel=idx, program=instrument)
                track.append(message)
                num_tracks += 1
                if num_tracks == 9:
                    num_tracks += 1 # skip the drums track

            track.append(mido.Message('note_on', note=pitch, channel=idx, velocity=96, time=time_in_ticks-previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)
        else:
            token -= MIDI_END_OFFSET
            instrument = token // 2**7
            pitch = token - (2**7)*instrument

            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                # shouldn't happen because we should have a corresponding onset
                if debug:
                    print('IGNORING bad offset')

                continue

            track.append(mido.Message('note_off', note=pitch, channel=idx, time=time_in_ticks-previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)

    return mid

def midi_to_compound_new(midifile, vocab, only_piano=False, harmonize=False, debug=False):
    # This function uses miditoolkit instead of mido objects to satisfy chorder's requirements

    harmonized = 0

    if type(midifile) == str:
        midi = miditoolkit.MidiFile(midifile)
    else:
        raise ValueError('midi_to_compound() requires a filepath to a midi file')
    
    programs = [i.program for i in midi.instruments]
    if harmonize and (vocab['chord_instrument'] - vocab['instrument_offset']) in programs:
        raise ValueError('Chord instrument already in midi file')
    
    if only_piano:
        # check for exactly one program 0
        program_zero_count = 0
        for instrument in midi.instruments:
            if instrument.program == 0 and len(instrument.notes) > 0 and not instrument.is_drum:
                program_zero_count += 1

        if program_zero_count != 1:
            raise ValueError("Each file must have exactly one instrument with program number 0.")
        
    time_res = vocab['config']['midi_quantization']
    # midi.ticks_per_beat = time_res

    # make max_ticks safe
    midi.max_tick = max([max([n.end for n in i.notes]) for i in midi.instruments])

    # make tempo changes safe
    midi.tempo_changes = [tc for tc in midi.tempo_changes if tc.time < midi.max_tick]
    
    # make time signature changes safe   
    midi.time_signature_changes = [ts for ts in midi.time_signature_changes if ts.time < midi.max_tick]

    # would a better cutoff be derived from MAX_TRACK_TIME_IN_SECONDS?
    if midi.max_tick > 1e7:
        raise ValueError

    if len(midi.time_signature_changes) > 200: # too hard to harmonize
        raise ValueError

    if harmonize:
        mtk_midi_copy = deepcopy(midi)
        
        if only_piano:
            # remove piano before harmonizing
            mtk_midi_copy.instruments = [i for i in midi.instruments if i.program != 0]

        # add chords as markers
        mtk_midi_enchord = Dechorder.enchord(mtk_midi_copy)
        # convert markers to midi notes
        mtk_midi_chords = play_chords(mtk_midi_enchord)
        # change instrument to midi instrument
        mtk_midi_chords.instruments[0].program = vocab['chord_instrument'] - vocab['instrument_offset']
        if len(mtk_midi_chords.instruments[0].notes) > 0:
            harmonized = 1
        midi.instruments.extend(mtk_midi_chords.instruments)
        # update max_ticks in case chords extend beyond original track
        midi.max_tick = max([max([n.end for n in i.notes]) for i in midi.instruments])

    tokens = []
    ticks_to_sec_map = midi.get_tick_to_time_mapping()

    for inst in midi.instruments:
        for note in inst.notes:
            # sanity check: negative time?
            if note.start < 0:
                raise ValueError

            # special case: channel 9 corresponds to is.drum flag!
            instr = 128 if inst.is_drum else inst.program

            token = []

            token.append(floor(time_res * ticks_to_sec_map[note.start]))
            token.append(floor(time_res * ticks_to_sec_map[note.end - note.start]))
            token.append(note.pitch)
            assert (-1 <= instr < 129)
            token.append(instr)
            token.append(note.velocity)

            tokens.append(token)

            # Does not parse
            # - tempo
            # - time signature
            # - aftertouch, polytouch, pitchweel, sequencer_specific
            # - control_change
            # - track_name, text, end_of_track, lyrics, key_signature, marker, etc.
            # - channel_prefix
            # = midi_port, smpte_offset, sysex

    tokens.sort(key=lambda x: x[0])
    tokens = [ite for tk in tokens for ite in tk]
    return tokens, harmonized

def midi_to_compound(midifile, vocab, debug=False):
    time_res = vocab['config']['midi_quantization']

    if type(midifile) == str:
        midi = mido.MidiFile(midifile)
    else:
        raise ValueError('midi_to_compound() requires a filepath to a midi file') 

    tokens = []
    note_idx = 0
    open_notes = defaultdict(list)

    time = 0
    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat
    for message in midi:
        time += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ['note_on', 'note_off']:
            # special case: channel 9 is drums!
            instr = 128 if message.channel == 9 else instruments[message.channel]

            if message.type == 'note_on' and message.velocity > 0: # onset
                # time quantization
                time_in_ticks = round(time_res*time)

                # Our compound word is: (time, duration, note, instr, velocity)
                tokens.append(time_in_ticks) # 5ms resolution
                tokens.append(-1) # placeholder (we'll fill this in later)
                tokens.append(message.note)
                tokens.append(instr)
                tokens.append(message.velocity)

                open_notes[(instr,message.note,message.channel)].append((note_idx, time))
                note_idx += 1
            else: # offset
                try:
                    open_idx, onset_time = open_notes[(instr,message.note,message.channel)].pop(0)
                except IndexError:
                    if debug:
                        print('WARNING: ignoring bad offset')
                else:
                    duration_ticks = round(time_res*(time-onset_time))
                    tokens[5*open_idx + 1] = duration_ticks
                    #del open_notes[(instr,message.note,message.channel)]
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    unclosed_count = 0
    for _,v in open_notes.items():
        unclosed_count += len(v)

    if debug and unclosed_count > 0:
        print(f'WARNING: {unclosed_count} unclosed notes')
        print('  ', midifile)

    return tokens


def compound_to_midi(tokens, vocab, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = vocab['config']['midi_quantization'] // 2 # 2 beats/second at quarter=120

    it = iter(tokens)
    time_index = defaultdict(list)
    for _, (time_in_ticks,duration,note,instrument,velocity) in enumerate(zip(it,it,it,it,it)):
        time_index[(time_in_ticks,0)].append((note, instrument, velocity)) # 0 = onset
        time_index[(time_in_ticks+duration,1)].append((note, instrument, velocity)) # 1 = offset

    track_idx = {} # maps instrument to (track number, current time)
    num_tracks = 0
    for time_in_ticks, event_type in sorted(time_index.keys()):
        for (note, instrument, velocity) in time_index[(time_in_ticks, event_type)]:
            if event_type == 0: # onset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    idx = num_tracks
                    previous_time = 0
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128: # drums always go on channel 9
                        idx = 9
                        message = mido.Message('program_change', channel=idx, program=0)
                    else:
                        message = mido.Message('program_change', channel=idx, program=instrument)
                    track.append(message)
                    num_tracks += 1
                    if num_tracks == 9:
                        num_tracks += 1 # skip the drums track

                track.append(mido.Message(
                    'note_on', note=note, channel=idx, velocity=velocity,
                    time=time_in_ticks-previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)
            else: # offset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    # shouldn't happen because we should have a corresponding onset
                    if debug:
                        print('IGNORING bad offset')

                    continue

                track.append(mido.Message(
                    'note_off', note=note, channel=idx,
                    time=time_in_ticks-previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


def compound_to_events(tokens, vocab, stats=False):
    time_offset = vocab['time_offset']
    note_offset = vocab['note_offset']
    separator = vocab['separator']
    dur_offset = vocab['duration_offset']

    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    # remove velocities
    del tokens[4::5]

    # combine (note, instrument)
    assert all(-1 <= tok < 2**7 for tok in tokens[2::4])
    assert all(-1 <= tok < 129 for tok in tokens[3::4])
    tokens[2::4] = [separator if note == -1 else MAX_PITCH*instr + note
                    for note, instr in zip(tokens[2::4],tokens[3::4])]
    tokens[2::4] = [note_offset + tok for tok in tokens[2::4]]
    del tokens[3::4]

    # max duration cutoff and set unknown durations to 250ms
    truncations = sum([1 for tok in tokens[1::3] if tok >= MAX_DUR])
    tokens[1::3] = [TIME_RESOLUTION//4 if tok == -1 else min(tok, MAX_DUR-1)
                    for tok in tokens[1::3]]
    tokens[1::3] = [dur_offset + tok for tok in tokens[1::3]]

    assert min(tokens[0::3]) >= 0
    tokens[0::3] = [time_offset + tok for tok in tokens[0::3]]

    assert len(tokens) % 3 == 0

    if stats:
        return tokens, truncations

    return tokens


def events_to_compound(tokens, debug=False):
    tokens = unpad(tokens)

    control_offset = vocab['control_offset']
    time_offset = vocab['time_offset']
    duration_offset = vocab['duration_offset']
    note_offset = vocab['note_offset']
    separator = vocab['separator']

    # move all tokens to zero-offset for synthesis
    tokens = [tok - control_offset if tok >= control_offset and tok != separator else tok
              for tok in tokens]
    
    # remove type offsets
    tokens[0::3] = [tok - time_offset if tok != separator else tok for tok in tokens[0::3]]
    tokens[1::3] = [tok - duration_offset if tok != separator else tok for tok in tokens[1::3]]
    tokens[2::3] = [tok - note_offset if tok != separator else tok for tok in tokens[2::3]]

    offset = 0 # add max time from previous track for synthesis
    track_max = 0 # keep track of max time in track
    for j, (time,dur,note) in enumerate(zip(tokens[0::3],tokens[1::3],tokens[2::3])):
        if note == separator:
            offset += track_max
            track_max = 0
            if debug:
                print('Sequence Boundary')
        else:
            track_max = max(track_max, time+dur)
            tokens[3*j] += offset

    # strip sequence separators
    assert len([tok for tok in tokens if tok == separator]) % 3 == 0
    tokens = [tok for tok in tokens if tok != separator]

    assert len(tokens) % 3 == 0
    out = 5*(len(tokens)//3)*[0]
    out[0::5] = tokens[0::3]
    out[1::5] = tokens[1::3]
    out[2::5] = [tok - (2**7)*(tok//2**7) for tok in tokens[2::3]]
    out[3::5] = [tok//2**7 for tok in tokens[2::3]]
    out[4::5] = (len(tokens)//3)*[72] # default velocity
    
    assert max(out[1::5]) < vocab['config']['max_duration']
    assert max(out[2::5]) < vocab['config']['max_note']
    assert max(out[3::5]) < vocab['config']['max_instrument']
    assert all(tok >= 0 for tok in out)

    return out


def compound_to_mm(tokens, vocab, stats=False):
    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    time_res = vocab['config']['midi_quantization']
    max_duration = vocab['config']['max_duration']
    max_interarrival = vocab['config']['max_interarrival']

    rest = [time_offset+max_interarrival, vocab['rest'], vocab['rest'], dur_offset+max_interarrival]

    # remove velocities
    del tokens[4::5]

    mm_tokens = [None] * len(tokens)

    # sanity check and offset
    assert all(-1 <= tok < 2**7 for tok in tokens[2::4])
    assert all(-1 <= tok < 129 for tok in tokens[3::4])
    mm_tokens[1::4] = [instr_offset + tok for tok in tokens[3::4]]
    mm_tokens[2::4] = [pitch_offset + tok for tok in tokens[2::4]]

    # max duration cutoff and set unknown durations to 250ms
    truncations = sum([1 for tok in tokens[1::4] if tok >= max_duration])
    mm_tokens[3::4] = [dur_offset + time_res//4 if tok == -1 else dur_offset + min(tok, max_duration-1)
                       for tok in tokens[1::4]]

    # convert to interarrival times
    assert min(tokens[0::4]) >= 0
    offset = 0
    for idx in range(len(tokens) // 4):
        if idx == 0:
            previous_time = 0

        time = tokens[4*idx]
        ia = time - previous_time
        while ia > max_interarrival:
            # insert a rest
            mm_tokens[4*(idx+offset):4*(idx+offset)] = rest.copy()
            ia -= max_interarrival
            offset += 1

        mm_tokens[4*(idx+offset)] = time_offset + ia
        previous_time = time

    if stats:
        return mm_tokens, truncations

    return mm_tokens


def make_events_safe(input_events):
    """
    Adjusts durations in an events list to prevent overlapping notes for each instrument.
    Events are triplets of (time, duration, note) tokens.
    Returns a new events list with adjusted durations.
    """
    # Create a copy of events list
    events = input_events.copy()
    
    # Group events by note (which encodes both pitch and instrument)
    note_events = {}
    for i in range(0, len(events), 3):
        time = events[i]
        dur = events[i+1] 
        note = events[i+2]
        
        if note not in note_events:
            note_events[note] = []
        note_events[note].append((i, time, dur))

    # For each note, check and fix overlaps
    for note, note_list in note_events.items():
        # Sort by time
        sorted_events = sorted(note_list, key=lambda x: x[1])
        
        # Check consecutive pairs
        for j in range(len(sorted_events)-1):
            curr_idx, curr_time, curr_dur = sorted_events[j]
            next_idx, next_time, _ = sorted_events[j+1]
            
            # Convert to absolute time by removing offsets
            # For control tokens, subtract both control offset and regular offset
            if note >= vocab['control_offset']:
                curr_abs_time = curr_time - vocab['control_offset'] - vocab['time_offset']
                next_abs_time = next_time - vocab['control_offset'] - vocab['time_offset']
                curr_abs_dur = curr_dur - vocab['control_offset'] - vocab['duration_offset']
            else:
                curr_abs_time = curr_time - vocab['time_offset']
                next_abs_time = next_time - vocab['time_offset']
                curr_abs_dur = curr_dur - vocab['duration_offset']
            
            # Check for overlap
            if curr_abs_time + curr_abs_dur >= next_abs_time:
                # Adjust duration to end 1 tick before next note
                new_abs_dur = next_abs_time - curr_abs_time - 1
                # Update duration token in new events list with appropriate offset(s)
                if note >= vocab['control_offset']:
                    events[curr_idx + 1] = vocab['control_offset'] + vocab['duration_offset'] + new_abs_dur
                else:
                    events[curr_idx + 1] = vocab['duration_offset'] + new_abs_dur

    return events


def events_to_midi(tokens, vocab, debug=False):
    return compound_to_midi(events_to_compound(tokens, debug=debug), vocab, debug=debug)

def midi_to_events(midifile, debug=False):
    return compound_to_events(midi_to_compound(midifile, vocab, debug=debug), vocab)

def midi_to_events_new(midifile, debug=False):
    return compound_to_events(midi_to_compound_new(midifile, vocab, debug=debug)[0], vocab)

def midi_to_mm(midifile, vocab, debug=False):
    return compound_to_mm(midi_to_compound(midifile, vocab, debug=debug), vocab)
