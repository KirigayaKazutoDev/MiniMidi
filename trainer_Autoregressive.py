import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from music21 import converter, note, stream, tempo, meter, chord


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        # sequences: list of (seq_len, 2) arrays
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # input: all but last, target: last step
        return seq[:-1], seq[1:]


def find_midi_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith('.mid') or f.lower().endswith('.midi'):
                files.append(os.path.join(root, f))
    return files


def extract_note_pairs(midi_path, max_notes_per_file=2000):
    pairs = []
    try:
        midi = converter.parse(midi_path)
        for elem in midi.flatten().notes:
            if isinstance(elem, note.Note):
                p = float(elem.pitch.midi) / 127.0
                d = float(elem.quarterLength) / 4.0
                pairs.append([p, d])
            elif isinstance(elem, chord.Chord):
                p = float(elem.pitches[0].midi) / 127.0
                d = float(elem.quarterLength) / 4.0
                pairs.append([p, d])
            if len(pairs) >= max_notes_per_file:
                break
    except Exception:
        return []
    return pairs


def build_sequences(all_pairs, seq_len, stride=1):
    sequences = []
    arr = np.array(all_pairs, dtype=np.float32)
    if len(arr) < seq_len + 1:
        return sequences
    for start in range(0, len(arr) - seq_len, stride):
        seq = arr[start:start + seq_len + 1]
        sequences.append(seq)
    return sequences


class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden=None):
        # x: (B, T, 2)
        out, hidden = self.lstm(x, hidden)
        return self.out(out), hidden


def detoken_to_stream(token_seq, pitch_offset=0):
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=120))
    p.append(meter.TimeSignature('4/4'))
    for t in token_seq:
        pitch_f, dur_f = t
        midi_pitch = int(np.clip(round(pitch_f * 127.0) + int(pitch_offset), 0, 127))
        # quantize duration to nearest 0.25
        q = max(0.25, round(dur_f * 4.0) / 4.0)
        n = note.Note(midi_pitch)
        n.quarterLength = q
        p.append(n)
    s.append(p)
    return s


def save_stream_as_musicxml(s, filename):
    s.write('musicxml', filename)


def human_readable_size(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def main():
    parser = argparse.ArgumentParser(description='Autoregressive LSTM MIDI prototype (regression on pitch+dur)')
    parser.add_argument('--folder', type=str, default='midi_files', help='MIDI folder (recursive)')
    parser.add_argument('--cut-data', type=int, default=0, help='Max number of files to load (0=all)')
    parser.add_argument('--cut-data-mode', type=str, default='first', choices=['first','alpha','largest','random'], help='How to pick files when using --cut-data: first, alpha, largest, random')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length (steps)')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window stride')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--save-path', type=str, default='generator_autoregressive.pt', help='Save path for generator')
    parser.add_argument('--gen-length', type=int, default=64, help='Length of sequence to generate')
    parser.add_argument('--pitch-offset', type=int, default=0, help='MIDI pitch offset applied on export')
    parser.add_argument('--seed-from-data', action='store_true', help='Seed generation from a real sequence')
    args = parser.parse_args()

    files = find_midi_files(args.folder)
    if not files:
        print('No MIDI files found.', flush=True)
        return

    # select candidate files according to --cut-data and --cut-data-mode
    candidate_files = files
    if args.cut_data and args.cut_data > 0:
        mode = args.cut_data_mode
        if mode == 'first':
            candidate_files = files[:args.cut_data]
        elif mode == 'alpha':
            candidate_files = sorted(files)[:args.cut_data]
        elif mode == 'largest':
            # sort by file size (descending), take top N
            candidate_files = sorted(files, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)[:args.cut_data]
        elif mode == 'random':
            import random
            candidate_files = files.copy()
            random.shuffle(candidate_files)
            candidate_files = candidate_files[:args.cut_data]
    else:
        candidate_files = files

    all_pairs = []
    file_count = 0
    total_bytes = 0
    for f in candidate_files:
        pairs = extract_note_pairs(f)
        if pairs:
            all_pairs.extend(pairs)
            file_count += 1
            try:
                if os.path.exists(f):
                    total_bytes += os.path.getsize(f)
            except Exception:
                pass

    if len(all_pairs) < args.seq_len + 1:
        print('Not enough note events to build sequences. Reduce --seq-len or provide more data.', flush=True)
        return

    sequences = build_sequences(all_pairs, args.seq_len, stride=args.stride)
    print(f'Built {len(sequences)} sequences from {file_count} files (total size: {human_readable_size(total_bytes)})', flush=True)

    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoregressiveLSTM(input_size=2, hidden_size=args.hidden, num_layers=args.num_layers).to(device)
    print(f'Autoregressive LSTM: hidden={args.hidden}, num_layers={args.num_layers}', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)  # (B, T-1, 2)
            yb = yb.to(device)  # (B, T-1, 2)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataset)
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {avg:.6f}', flush=True)

    try:
        torch.save(model.state_dict(), args.save_path)
        print(f'Saved generator to {args.save_path}', flush=True)
    except Exception as e:
        print(f'Warning: could not save generator: {e}', flush=True)

    # Generation
    model.eval()
    with torch.no_grad():
        if args.seed_from_data:
            # pick random sequence from dataset
            seed_idx = np.random.randint(0, len(sequences))
            seed = torch.tensor(sequences[seed_idx][:args.seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            # random seed from data distribution
            seed_np = np.array(all_pairs[:args.seq_len], dtype=np.float32)
            seed = torch.tensor(seed_np, dtype=torch.float32).unsqueeze(0).to(device)

        generated = seed.squeeze(0).cpu().numpy().tolist()
        input_seq = seed
        hidden = None
        for _ in range(args.gen_length):
            pred, hidden = model(input_seq, hidden)
            # take last step prediction
            last = pred[:, -1, :].cpu().numpy()[0]
            # append and shift input_seq
            generated.append(last.tolist())
            # build new input_seq tensor: take last (seq_len) of generated
            arr = np.array(generated[-args.seq_len:]).astype(np.float32)
            input_seq = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

    # export generated sequence to musicxml
    s = detoken_to_stream(generated, pitch_offset=args.pitch_offset)
    out_path = Path('generated_autoregressive.musicxml')
    save_stream_as_musicxml(s, str(out_path))

    # report generation stats
    generated_len = len(generated)
    # quantize durations same as in detoken_to_stream
    quantized_durs = [max(0.25, round(t[1] * 4.0) / 4.0) for t in generated]
    total_quarters = sum(quantized_durs)
    print(f'gen-length requested: {args.gen_length}', flush=True)
    print(f'seed length (seq-len): {args.seq_len}', flush=True)
    print(f'total tokens exported: {generated_len}', flush=True)
    print(f'Estimated total duration: {total_quarters} quarter notes ({total_quarters/4.0} bars in 4/4)', flush=True)
    print(f'Exported generated sequence to {out_path}', flush=True)


if __name__ == '__main__':
    main()
