import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def visualize_midi_tracks(data):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    track_height = 1
    track_spacing = 0.2
    y_offset = 0
    end_time = max(track['clips'][-1]['end_time'] for track in data if track['clips'])
    
    for track in data:
        track_color = generate_random_color()
        track_name = track['name'][0]
        
        # Draw track rectangle
        ax.add_patch(patches.Rectangle((0, y_offset), end_time, track_height, fill=False, edgecolor=track_color))
        ax.text(-0.1, y_offset + track_height/2, track_name, verticalalignment='center', horizontalalignment='right')
        
        for clip in track['clips']:
            clip_start = clip['start_time']
            clip_end = clip['end_time']
            clip_duration = clip_end - clip_start
            clip_color = generate_random_color()
            
            # Draw clip rectangle
            
            ax.add_patch(patches.Rectangle((clip_start, y_offset), clip_duration, track_height, fill=True, facecolor=clip_color, alpha=0.5))
            ax.text(clip_start + clip_duration/2, y_offset + track_height*.9, clip['name'][0], horizontalalignment='center', verticalalignment='center')

            if len(clip['notes']) == 0:
                ax.add_patch(patches.Rectangle((clip_start, y_offset), clip_duration, track_height, fill=False, alpha=0.5, hatch='//'))
                ax.text(clip_start + clip_duration/2, y_offset + track_height/2, "EMPTY", horizontalalignment='center', verticalalignment='center')

            

            
            for note in clip['notes']:
                note_start = note['start_time']
                note_duration = note['duration']
                note_pitch = note['pitch']
                
                # Normalize pitch to track height
                note_y = y_offset + (note_pitch / 127) * track_height
                
                # Draw note rectangle
                ax.add_patch(patches.Rectangle((clip_start + note_start, note_y), note_duration, 0.02, fill=True, facecolor='black'))         
        
        y_offset += track_height + track_spacing

    ax.set_yticks([])
    ax.set_xlim(0, max(track['clips'][-1]['end_time'] for track in data if track['clips']))
    ax.set_ylim(0, y_offset)
    ax.set_xlabel('Time')
    ax.set_ylabel('Tracks')
    ax.set_title('Ableton Live MIDI Arrangement State')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to your JSON file as an argument.")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_json(file_path)
    visualize_midi_tracks(data)