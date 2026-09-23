import os
import sys
import mmap
import msvcrt  # Built-in Windows console input module

TEXT_FILE = "model_raw_weights.txt"
LINES_PER_PAGE = 30  # Adjust based on your terminal height

def get_terminal_size():
    try:
        columns, lines = os.get_terminal_size()
        return lines - 3  # Reserve 3 lines for status bar and header
    except OSError:
        return LINES_PER_PAGE

def render_page(lines_buffer, current_line_idx, total_lines):
    # Clear screen on Windows console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 80)
    print(f" MODEL RAW WEIGHTS VIEWER | File: {TEXT_FILE}")
    print(" Navigation: [UP/DOWN] Line | [PGUP/PGDN/SPACE] Page | [HOME/END] Jump | [Q] Quit")
    print("=" * 80)
    
    for line in lines_buffer:
        # Truncate overly long tensor lines so they fit cleanly on screen
        print(line[:120] if len(line) > 120 else line)

    print("-" * 80)
    print(f" Line {current_line_idx + 1:,} of ~{total_lines:,} | Press 'Q' to exit")

def build_line_index(filename):
    """Scans file once to build line offsets for O(1) instant seeking without loading content into RAM."""
    print(f"[+] Indexing '{filename}' for instant seeking (memory-mapped)...")
    line_offsets = [0]
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            while mm.readline():
                line_offsets.append(mm.tell())
    # Remove the final EOF offset
    line_offsets.pop()
    return line_offsets

def main():
    if not os.path.exists(TEXT_FILE):
        print(f"[-] Error: '{TEXT_FILE}' not found. Please run the exporter script first.")
        sys.exit(1)

    line_offsets = build_line_index(TEXT_FILE)
    total_lines = len(line_offsets)
    current_line = 0

    with open(TEXT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            while True:
                page_size = get_terminal_size()
                
                # Seek directly to the exact byte offset of the current line
                mm.seek(line_offsets[current_line])
                
                # Read page_size lines
                lines_buffer = []
                for _ in range(page_size):
                    raw_line = mm.readline()
                    if not raw_line:
                        break
                    lines_buffer.append(raw_line.decode("utf-8", errors="ignore").rstrip("\r\n"))

                render_page(lines_buffer, current_line, total_lines)

                # Capture Windows Key Press via msvcrt
                key = msvcrt.getch()

                if key in (b'q', b'Q'):
                    break
                elif key == b' ':  # Space bar -> Page Down
                    current_line = min(current_line + page_size, max(0, total_lines - page_size))
                elif key in (b'\x00', b'\xe0'):  # Special key prefix on Windows
                    nav_key = msvcrt.getch()
                    
                    if nav_key == b'H':    # Up Arrow
                        current_line = max(0, current_line - 1)
                    elif nav_key == b'P':  # Down Arrow
                        current_line = min(total_lines - 1, current_line + 1)
                    elif nav_key == b'I':  # Page Up
                        current_line = max(0, current_line - page_size)
                    elif nav_key == b'Q':  # Page Down
                        current_line = min(total_lines - 1, current_line + page_size)
                    elif nav_key == b'G':  # Home
                        current_line = 0
                    elif nav_key == b'O':  # End
                        current_line = max(0, total_lines - page_size)

    print("\n[+] Viewer closed.")

if __name__ == "__main__":
    main()