import os
import argparse
import torch
import torch.nn as nn
from music21 import converter, note, chord, stream, tempo, meter

# 1. Argumente & Hardware-Setup
parser = argparse.ArgumentParser(description="MIDI GAN mit Modell-Speicherung")
parser.add_argument("--use-fp16", action="store_true", help="Aktiviert FP16 Mixed Precision auf CUDA")
parser.add_argument("--pretrain", action="store_true", help="Führt das Pretraining des Evaluators aus und speichert ihn")
parser.add_argument("--evaluator-path", type=str, default="evaluator.pt", help="Pfad zum gespeicherten Evaluator-Modell")
parser.add_argument("--cut-data", type=int, default=0, help="Optional: max. Anzahl MIDI-Dateien laden (0 = alle)")
parser.add_argument("--cut-data-mode", type=str, default='first', choices=['first','alpha','largest','random'], help='How to pick files when using --cut-data: first, alpha, largest, random')
parser.add_argument("--generator-path", type=str, default="generator.pt", help="Pfad zum gespeicherten Generator-Modell")
parser.add_argument("--gen-steps", type=int, default=500, help="Anzahl Trainingsschritte für den Generator")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = args.use_fp16 and device.type == 'cuda'
# Use the stable `torch.amp` API (device-aware)
scaler = torch.amp.GradScaler(enabled=use_amp, device=device)

# 2. MIDI-Daten laden (nur beim Pretraining relevant)
def load_real_midi_data(folder_path="midi_files", max_notes=2000, max_files=0):
    real_notes = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Ordner '{folder_path}' wurde erstellt. Bitte lege dort .mid/.midi Dateien ab!", flush=True)
        return None

    # collect all MIDI files recursively
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.mid') or f.lower().endswith('.midi'):
                all_files.append(os.path.join(root, f))

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

    file_count = 0
    total_bytes = 0
    for path in candidate_files:
        try:
            midi = converter.parse(path)
            for elem in midi.flatten().notes:
                if isinstance(elem, note.Note):
                    p = elem.pitch.midi / 128.0
                    d = float(elem.quarterLength) / 4.0
                    real_notes.append([p, d])
                elif isinstance(elem, chord.Chord):
                    p = elem.pitches[0].midi / 128.0
                    d = float(elem.quarterLength) / 4.0
                    real_notes.append([p, d])
                if len(real_notes) >= max_notes:
                    break
            if len(real_notes) > 0:
                file_count += 1
                try:
                    if os.path.exists(path):
                        total_bytes += os.path.getsize(path)
                except Exception:
                    pass
        except Exception as e:
            print(f"Fehler beim Lesen von {path}: {e}")

    if not real_notes:
        return None

    # print size/summary for used files
    def human_readable_size(nbytes: int) -> str:
        n = float(nbytes)
        for unit in ['B','KiB','MiB','GiB']:
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} TiB"

    print(f"Loaded {len(real_notes)} note events from {file_count} files (total size: {human_readable_size(total_bytes)})", flush=True)
    return torch.tensor(real_notes, dtype=torch.float32)

# 3. Netzwerke
class MidiGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, z):
        return self.net(z)

class MidiEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

generator = MidiGenerator().to(device)
evaluator = MidiEvaluator().to(device)

optimizer_E = torch.optim.Adam(evaluator.parameters(), lr=0.001)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 4. Pretraining-Funktion
def pretrain_and_save(real_data, save_path, epochs=300, batch_size=64):
    print("\n--- PHASE 1: Start Pretraining ---", flush=True)
    evaluator.train()
    dataset_size = len(real_data)
    
    for epoch in range(epochs):
        idx = torch.randint(0, dataset_size, (batch_size,))
        real_batch = real_data[idx].to(device)
        fake_batch = torch.rand((batch_size, 2), device=device)
        
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
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss.item():.4f}", flush=True)

    # Speichert die Gewichte auf der Festplatte
    torch.save(evaluator.state_dict(), save_path)
    print(f"Modell erfolgreich gespeichert unter: {save_path}\n", flush=True)

# 5. Haupt-Ablauf
if __name__ == '__main__':
    # FALL 1: Pretraining explizit gefordert
    if args.pretrain:
        real_data = load_real_midi_data(max_files=args.cut_data)
        if real_data is not None:
            pretrain_and_save(real_data, args.evaluator_path)
        else:
            print("Abbruch: Keine MIDI-Dateien für das Pretraining gefunden.", flush=True)
            exit()
            
    # FALL 2: Bestehendes Modell laden
    else:
        if os.path.exists(args.evaluator_path):
            evaluator.load_state_dict(torch.load(args.evaluator_path, map_location=device))
            print(f"Erfolgreich geladenes Evaluator-Modell: {args.evaluator_path}")
        else:
            print(f"Fehler: Datei '{args.evaluator_path}' nicht gefunden.")
            print("Starte das Skript einmalig mit '--pretrain', um das Modell zu generieren.")
            exit()

    # PHASE 2: Generator trainieren
    print("\n--- PHASE 2: Generator Training ---", flush=True)
    evaluator.eval()
    generator.train()
    print(f"Generator-Training: {args.gen_steps} Schritte, Batch-Größe=64", flush=True)
    for step in range(args.gen_steps):
        noise = torch.randn(64, 16, device=device)
        optimizer_G.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            gen_notes = generator(noise)
            scores = evaluator(gen_notes)
            loss_G = criterion(scores, torch.ones_like(scores))
            
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()
    # Speichere den Generator nach dem Training
    try:
        torch.save(generator.state_dict(), args.generator_path)
        print(f"Generator gespeichert unter: {args.generator_path}", flush=True)
    except Exception as e:
        print(f"Warnung: Generator konnte nicht gespeichert werden: {e}", flush=True)
    print("Generator fertig trainiert basierend auf dem geladenen Evaluator!", flush=True)