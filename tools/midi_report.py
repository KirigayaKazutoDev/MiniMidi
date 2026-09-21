import os
import argparse
from pathlib import Path


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Ei{suffix}"


def find_midi_files(folder_path):
    midi_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.mid') or f.lower().endswith('.midi'):
                full = os.path.join(root, f)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                midi_files.append((full, size))
    return midi_files


def write_report(folder_path, out_path, top_n=100):
    folder = Path(folder_path)
    out = Path(out_path)
    midi_files = find_midi_files(folder)
    total_files = len(midi_files)
    total_bytes = sum(s for _, s in midi_files)

    midi_files.sort(key=lambda x: x[1], reverse=True)
    top = midi_files[:top_n]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        f.write(f"MIDI Report for: {folder.resolve()}\n")
        f.write(f"Total MIDI files: {total_files}\n")
        f.write(f"Total size: {total_bytes} bytes ({sizeof_fmt(total_bytes)})\n")
        f.write('\n')
        f.write(f"Top {len(top)} largest MIDI files:\n")
        for i, (path, size) in enumerate(top, start=1):
            rel = Path(path).resolve()
            pct = (size / total_bytes * 100) if total_bytes > 0 else 0
            f.write(f"{i:3d}. {sizeof_fmt(size):>10}  {pct:6.2f}%  {rel}\n")

    return out.resolve()


def main():
    parser = argparse.ArgumentParser(description='Count MIDI files and list largest ones')
    parser.add_argument('--folder', type=str, default=r'midi_files\\archive', help='Folder to scan')
    parser.add_argument('--output', type=str, default=None, help='Output txt file path')
    parser.add_argument('--top', type=int, default=100, help='How many largest files to list')
    args = parser.parse_args()

    folder = args.folder
    if args.output:
        out = args.output
    else:
        out = os.path.join(folder, 'midi_report.txt')

    report_path = write_report(folder, out, top_n=args.top)
    print(f"Report written to: {report_path}")


if __name__ == '__main__':
    main()
