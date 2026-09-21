import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from music21 import stream, note, chord, tempo, meter, converter

# 1. Parameter & CLI Setup
parser = argparse.ArgumentParser(description="Piano-Roll MIDI GAN (Akkorde & Mehrstimmigkeit)")
parser.add_argument("--use-fp16", action="store_true", help="Aktiviert FP16 Mixed Precision auf CUDA")
parser.add_argument("--pretrain", action="store_true", help="Führt Pretraining des Evaluators aus")
parser.add_argument("--evaluator-path", type=str, default="evaluator_pianoroll.pt", help="Pfad zum gespeicherten Evaluator")
parser.add_argument("--cut-data", type=int, default=0, help="Optional: max. Anzahl MIDI-Dateien laden (0 = alle)")
parser.add_argument("--cut-data-mode", type=str, default='first', choices=['first','alpha','largest','random'], help='How to pick files when using --cut-data: first, alpha, largest, random')
parser.add_argument("--generator-path", type=str, default="generator_pianoroll.pt", help="Pfad zum gespeicherten Generator")
parser.add_argument("--gen-steps", type=int, default=1000, help="Anzahl Trainingsschritte für den Generator")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = args.use_fp16 and device.type == 'cuda'
# Use the stable `torch.amp` API (device-aware)
scaler = torch.amp.GradScaler(enabled=use_amp, device=device)

# Grid-Konfiguration: 16 Zeitschritte (1 Takt), 36 Tonhöhen (Midi 48 bis 83 -> C3 bis B5)
STEPS = 16
PITCHES = 36
PITCH_OFFSET = 48 

# 2. MIDI zu Piano-Roll Tensor konvertieren
def midi_to_pianoroll(file_path):
    grid = np.zeros((STEPS, PITCHES), dtype=np.float32)
    try:
        midi = converter.parse(file_path)
        # Alle Noten & Akkorde flach auslesen
        elements = midi.flatten().notes
        for elem in elements:
            start_step = int(elem.offset * 4) # 1 Viertel = 4 Steps
            duration_steps = max(1, int(elem.quarterLength * 4))
            end_step = min(STEPS, start_step + duration_steps)
            
            if start_step >= STEPS:
                continue

            # Einzelnote
            if isinstance(elem, note.Note):
                p_idx = elem.pitch.midi - PITCH_OFFSET
                if 0 <= p_idx < PITCHES:
                    grid[start_step:end_step, p_idx] = 1.0
            # Akkord (Alle Töne verarbeiten!)
            elif isinstance(elem, chord.Chord):
                for p in elem.pitches:
                    p_idx = p.midi - PITCH_OFFSET
                    if 0 <= p_idx < PITCHES:
                        grid[start_step:end_step, p_idx] = 1.0
    except Exception as e:
        return None
    return grid

def load_all_pianorolls(folder_path="midi_files", max_files=0):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Ordner '{folder_path}' wurde erstellt. Lege dort deine .mid/.midi Dateien ab!", flush=True)
        return None
    dataset = []
    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                all_files.append(os.path.join(root, file))

    # Select candidate files according to mode
    if max_files and max_files > 0:
        mode = args.cut_data_mode if hasattr(args, 'cut_data_mode') else 'first'
        if mode == 'first':
            candidate_files = all_files[:max_files]
        elif mode == 'alpha':
            candidate_files = sorted(all_files)[:max_files]
        elif mode == 'largest':
            candidate_files = sorted(all_files, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)[:max_files]
        elif mode == 'random':
            import random
            candidate_files = all_files.copy()
            random.shuffle(candidate_files)
            candidate_files = candidate_files[:max_files]
    else:
        candidate_files = all_files

    total_bytes = 0
    for path in candidate_files:
        grid = midi_to_pianoroll(path)
        if grid is not None and np.sum(grid) > 0:
            dataset.append(grid)
            try:
                if os.path.exists(path):
                    total_bytes += os.path.getsize(path)
            except Exception:
                pass

    if not dataset:
        return None

    def human_readable_size(nbytes: int) -> str:
        n = float(nbytes)
        for unit in ['B','KiB','MiB','GiB']:
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} TiB"

    print(f'Loaded {len(dataset)} piano-rolls (total size: {human_readable_size(total_bytes)})', flush=True)

    # Form: (Anzahl_Songs, 1, 16, 36) -> 2D-Grid für Conv2D
    return torch.tensor(np.array(dataset), dtype=torch.float32).unsqueeze(1)

# 3. Netzwerke mit Convolutional Layers (ideal für 2D Piano-Rolls)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, STEPS * PITCHES),
            nn.Sigmoid() # Werte zwischen 0.0 und 1.0 (Wahrscheinlichkeiten)
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, 1, STEPS, PITCHES)

class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(STEPS * PITCHES, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
evaluator = Evaluator().to(device)

optimizer_E = torch.optim.Adam(evaluator.parameters(), lr=0.0002)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# 4. Pretraining & Training Loop
def pretrain_evaluator(real_data, epochs=300, batch_size=16):
    print("\n--- Pretraining Evaluator auf Piano-Rolls (Akkorde & Mehrstimmigkeit) ---", flush=True)
    evaluator.train()
    dataset_size = len(real_data)
    
    for epoch in range(epochs):
        idx = torch.randint(0, dataset_size, (min(batch_size, dataset_size),))
        real_batch = real_data[idx].to(device)
        fake_batch = torch.rand_like(real_batch, device=device)
        
        optimizer_E.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_real = evaluator(real_batch)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))
            pred_fake = evaluator(fake_batch)
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            total_loss = loss_real + loss_fake
            
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_E)
        scaler.update()

    torch.save(evaluator.state_dict(), args.evaluator_path)
    print(f"Evaluator gespeichert unter {args.evaluator_path}\n", flush=True)

# 5. Piano-Roll zurück in MuseScore MusicXML konvertieren
def export_pianoroll_to_musescore(grid_tensor, filename="piano_roll_output.musicxml"):
    # Schwellenwert: Alles über 0.5 wird als aktivierte Note behandelt
    grid = (grid_tensor.detach().squeeze().cpu().numpy() > 0.5).astype(int)
    
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=120))
    p.append(meter.TimeSignature('4/4'))

    for step in range(STEPS):
        active_pitches = np.where(grid[step] == 1)[0]
        
        if len(active_pitches) == 1:
            # Einzelnote
            n = note.Note(int(active_pitches[0] + PITCH_OFFSET))
            n.quarterLength = 0.25 # Sechzehntelnote
            p.append(n)
        elif len(active_pitches) > 1:
            # Akkord (Mehrere Töne zeitgleich!)
            c = chord.Chord([int(p_idx + PITCH_OFFSET) for p_idx in active_pitches])
            c.quarterLength = 0.25
            p.append(c)

    s.append(p)
    s.write('musicxml', filename)
    print(f"Partitur mit Akkorden exportiert als: {filename}", flush=True)

if __name__ == '__main__':
    if args.pretrain:
        real_data = load_all_pianorolls(max_files=args.cut_data)
        if real_data is not None:
            pretrain_evaluator(real_data)
        else:
            print("Keine verarbeitbaren MIDI-Dateien in 'midi_files' gefunden.", flush=True)
            exit()
    else:
        if os.path.exists(args.evaluator_path):
            evaluator.load_state_dict(torch.load(args.evaluator_path, map_location=device))
            print(f"Evaluator geladen von: {args.evaluator_path}", flush=True)
        else:
            print(f"Datei '{args.evaluator_path}' nicht gefunden. Starte erst mit '--pretrain'.", flush=True)
            exit()

    # Generator Training
    print("--- Generiere Akkorde & Melodien ---", flush=True)
    evaluator.eval()
    generator.train()
    print(f"Generator-Training: {args.gen_steps} Schritte, Batch-Größe=16", flush=True)
    for step in range(args.gen_steps):
        noise = torch.randn(16, 32, device=device)
        optimizer_G.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            gen_pianorolls = generator(noise)
            scores = evaluator(gen_pianorolls)
            loss_G = criterion(scores, torch.ones_like(scores))

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

    # Letztes Ergebnis exportieren
    # Save generator weights and export one sample
    try:
        torch.save(generator.state_dict(), args.generator_path)
        print(f"Generator gespeichert unter {args.generator_path}", flush=True)
    except Exception:
        print("Warnung: Generator konnte nicht gespeichert werden.", flush=True)
    export_pianoroll_to_musescore(gen_pianorolls[0])