import mido
import matplotlib.pyplot as plt
import time
from collections import deque

# Define MIDI note labels for Y-axis
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
midi_range = range(36, 97)  # C2 (36) to C7 (96)
y_labels = [f"{note_names[n % 12]}{(n // 12) - 1}" for n in midi_range]  # Create labels for all notes

# Data storage for real-time visualization
note_events = deque(maxlen=100)
timestamps = deque(maxlen=100)
note_status = {}

# Set up the piano roll graph
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel("Time (s)")
ax.set_ylabel("MIDI Note")
ax.set_title("Live MIDI Piano Roll")
ax.set_xlim(0, 10)
ax.set_ylim(min(midi_range), max(midi_range))
ax.set_yticks(list(midi_range))
ax.set_yticklabels(y_labels)
ax.grid(True)

# Detect and select MIDI device
fantom_midi_port = None
for port in mido.get_input_names():
    if "FANTOM" in port:
        fantom_midi_port = port
        break

if not fantom_midi_port:
    print("No FANTOM-06 MIDI device found. Make sure it is connected.")
    exit()

# Open MIDI input port
with mido.open_input(fantom_midi_port) as inport:
    print(f"Listening for MIDI notes from {fantom_midi_port}...")

    while True:
        # Read new MIDI messages
        for msg in inport.iter_pending():
            if msg.type in ["note_on", "note_off"]:
                current_time = time.time()

                if msg.type == "note_on" and msg.velocity > 0:
                    note_events.append(msg.note)
                    timestamps.append(current_time)
                    note_status[msg.note] = current_time
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    note_status.pop(msg.note, None)

        # Update plot even if no new notes are played
        current_time = time.time()
        ax.clear()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Note")
        ax.set_title("Live MIDI Piano Roll")
        ax.set_xlim(current_time - 10, current_time)  # Smooth scrolling
        ax.set_ylim(min(midi_range), max(midi_range))
        ax.set_yticks(list(midi_range))
        ax.set_yticklabels(y_labels)
        ax.grid(True)

        # Active and released note visualization
        active_notes = [n for n in note_events if n in note_status]
        released_notes = [n for n in note_events if n not in note_status]
        active_timestamps = [timestamps[i] for i, n in enumerate(note_events) if n in note_status]
        released_timestamps = [timestamps[i] for i, n in enumerate(note_events) if n not in note_status]

        ax.scatter(active_timestamps, active_notes, color="blue", marker="o", label="Active Notes")
        ax.scatter(released_timestamps, released_notes, color="red", marker="x", label="Released Notes")

        ax.legend()
        plt.pause(0.05)
