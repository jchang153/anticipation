"""
Utilities for inspecting encoded music data.
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import anticipation.ops as ops
from anticipation.config import *

def visualize(tokens, output, vocab, selected=None):
    #colors = ['white', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan',
    #          'dodgerblue', 'slategray', 'navy', 'mediumpurple', 'mediumorchid', 'magenta', 'lightpink']
    colors = ['white', '#426aa0', '#b26789', '#de9283', '#eac29f', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan', 'dodgerblue', 'slategray', 'navy']

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    controls_present = False
    max_time = ops.max_time_including_duration(tokens, seconds=False)
    grid = np.zeros([max_time, MAX_PITCH])
    control_grid = np.zeros([max_time, MAX_PITCH], dtype=bool)
    instruments = list(sorted(list(ops.get_instruments(tokens).keys())))
    if 128 in instruments:
        instruments.remove(128)
 
    for j, (tm, dur, note) in enumerate(zip(tokens[0::3],tokens[1::3],tokens[2::3])):
        # This separator token parsing should be deprecated with most recent control prefix.
        # Assume tokens has prefix removed for now.
        if note == vocab['separator']:
            assert tm == vocab['separator'] and dur == vocab['separator']
            print(j, 'SEPARATOR')
            continue

        if note == vocab['rest']:
            continue

        is_control = note >= vocab['control_offset']
        
        if not is_control:
            tm = tm - vocab['time_offset']
            dur = dur - vocab['duration_offset']
            note = note - vocab['note_offset']
        else:
            tm = tm - vocab['time_offset'] - vocab['control_offset']
            dur = dur - vocab['duration_offset'] - vocab['control_offset']
            note = note - vocab['note_offset'] - vocab['control_offset']
        
        instr = note // 2**7
        pitch = note - (2**7) * instr
        
        if instr == 128:  # drums
            continue  # we don't visualize this
        if selected and instr not in selected:
            continue
        
        grid[tm:tm+dur, pitch] = 1 + instruments.index(instr)
        if is_control:
            controls_present = True
            control_grid[tm:tm+dur, pitch] = True

    plt.clf()
    fig, ax = plt.subplots()
    ax.axis('off')
    
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = list(range(MAX_TRACK_INSTR)) + [16]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the main grid
    ax.imshow(np.flipud(grid.T), aspect=16, cmap=cmap, norm=norm, interpolation='none')
    
    # Add outlines for control tokens
    for i in range(control_grid.shape[1]):
        for j in range(control_grid.shape[0]):
            if control_grid[j, i]:
                # rect = patches.Rectangle((j-0.5, control_grid.shape[1]-i-1.5), 1, 1, 
                                          # alpha=1, fill=False, edgecolor='red', linewidth=.05)
                # rect = patches.Rectangle((j-0.5, control_grid.shape[1]-i-1.5), 1, 1, 
                #                           alpha=0.45, linestyle='--', facecolor='black')
                # ax.add_patch(rect)

                top_edge =    plt.Line2D([j-0.5, j+0.5], [control_grid.shape[1]-i-1.5, control_grid.shape[1]-i-1.5], color='black', linewidth=1)
            
                ax.add_line(top_edge)


    legend_patches = [matplotlib.patches.Patch(color=colors[i+1], label=f"{instruments[i]}")
                      for i in range(len(instruments))]

    if controls_present:
        legend_patches.append(matplotlib.patches.Patch(facecolor='black', alpha=1, label='Control'))
    
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output)

