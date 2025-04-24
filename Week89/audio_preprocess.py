import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

# Configuration will be passed externally

# ========== Utility Functions ==========

def midi_to_note_array(midi_path):
    """
    Extract note information from a MIDI file.
    Input: Path to MIDI file
    Return: Numpy array with [start, end, pitch, velocity] for each note
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    note_array = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            note_array.append([
                note.start,
                note.end,
                note.pitch,
                note.velocity
            ])
    return np.array(note_array)

def split_overlapping_notes(note_array, epsilon=1e-3):
    """
    Adjust overlapping notes with same pitch to ensure they do not touch.
    Input: Note array [start, end, pitch, velocity], epsilon to separate
    Return: Adjusted note array
    """
    separated = []
    for i in range(len(note_array)):
        start, end, pitch, velocity = note_array[i]
        if i < len(note_array) - 1:
            next_start, _, next_pitch, _ = note_array[i + 1]
            if pitch == next_pitch and abs(end - next_start) < 1e-6:
                end = next_start - epsilon
        separated.append([start, end, pitch, velocity])
    return np.array(separated)

def note_array_to_midi_file(note_array, out_path="temp_fixed.mid"):
    """
    Convert a note array back into a MIDI file.
    Input: Note array, output path
    Return: Path to saved MIDI file
    """
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for start, end, pitch, velocity in note_array:
        piano.notes.append(pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start),
            end=float(end)
        ))
    midi.instruments.append(piano)
    midi.write(out_path)
    return out_path

def extract_cqt_and_pianoroll(audio_file, midi_file, sr, hop_length, fmin, n_bins, bins_per_octave, n_frames=1000):
    """
    Compute CQT from audio and aligned pianoroll from MIDI.
    Input: audio file path, midi file path, and CQT config parameters
    Return: CQT in dB (n_bins x time), Pianoroll (128 x time)
    """
    y, _ = librosa.load(audio_file, sr=sr)

    C = librosa.cqt(
        y, sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )
    C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    note_array = midi_to_note_array(midi_file)
    adjusted_note_array = split_overlapping_notes(note_array, epsilon=5e-2)
    temp_midi = f'temp_{os.path.basename(midi_file)}'
    fixed_midi = note_array_to_midi_file(adjusted_note_array, out_path=temp_midi)

    midi_data = pretty_midi.PrettyMIDI(fixed_midi)
    fs_pianoroll = sr / hop_length
    piano_roll = midi_data.get_piano_roll(fs=fs_pianoroll)

    return C_dB[:, :n_frames], piano_roll[:, :n_frames]

def generate_cqt_sliding_windows(cqt, window_size=9, stride=1):
    """
    Create sliding windows from CQT spectrogram.
    Input: CQT matrix (n_bins x time), window size, stride
    Return: Sliding window array (num_windows x n_bins x window_size x 1)
    """
    pad = window_size // 2
    cqt_padded = np.pad(cqt, ((0, 0), (pad, pad)), mode='edge')
    num_time_bins = cqt.shape[1]

    windows = np.array([
        cqt_padded[:, i:i + window_size]
        for i in range(0, num_time_bins, stride)
    ])

    return windows[..., np.newaxis]

def generate_midi_sliding_windows(midi, window_size=9, stride=1):
    """
    Create sliding windows from MIDI pianoroll.
    Input: Pianoroll (128 x time), window size, stride
    Return: Sliding windows, number of windows
    """
    pad = window_size // 2
    midi_padded = np.pad(midi, ((0, 0), (pad, pad)), mode='constant', constant_values=0)
    num_time_bins = midi.shape[1]

    windows = np.array([
        midi_padded[:, i:i + window_size]
        for i in range(0, num_time_bins, stride)
    ])

    return windows, len(windows)