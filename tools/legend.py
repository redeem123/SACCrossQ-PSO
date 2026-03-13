import os
import argparse
from pypdf import PdfReader, PdfWriter

def clip_pdf(input_path, output_path, left=0, bottom=0, right=0, top=0):
    """
    Clips the margins of a PDF file.
    
    Args:
        input_path (str): Path to the input PDF.
        output_path (str): Path to save the clipped PDF.
        left, bottom, right, top (float): Points to trim from each side.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return

    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()

        for page in reader.pages:
            mb = page.mediabox
            
            # Apply the crop by adjusting the Mediabox
            # Values are typically in points (1/72 inch)
            page.mediabox.left = float(mb.left) + left
            page.mediabox.bottom = float(mb.bottom) + bottom
            page.mediabox.right = float(mb.right) - right
            page.mediabox.top = float(mb.top) - top
            
            writer.add_page(page)

        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"Successfully clipped: {os.path.basename(input_path)}")
        print(f"  -> Path: {output_path}")
        print(f"  -> Trims: L={left}, B={bottom}, R={right}, T={top}")

    except Exception as e:
        print(f"Failed to clip {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clip PDF margins for UAV paper figures.")
    parser.add_argument("-d", "--dir", type=str, help="Directory containing figures")
    parser.add_argument("-l", "--left", type=float, help="Override points to trim from left")
    parser.add_argument("-b", "--bottom", type=float, help="Override points to trim from bottom")
    parser.add_argument("-r", "--right", type=float, help="Override points to trim from right")
    parser.add_argument("-t", "--top", type=float, help="Override points to trim from top")
    parser.add_argument("--suffix", type=str, default="_clipped", help="Suffix for output files (default: _clipped)")
    
    parser.add_argument("--inplace", action="store_true", help="Overwrite the original files instead of creating new ones")
    
    args = parser.parse_args()

    # Determine figure directory
    if args.dir:
        fig_dir = os.path.abspath(args.dir)
    else:
        # Default to the others directory where legend.pdf is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        fig_dir = os.path.join(repo_root, "paper", "data", "others3chrismast")

    # Define standard targets and their default trims (L, B, R, T)
    trims = {
        "legend.pdf": {"left": 666, "bottom": 413, "right": 89, "top": 50},
    }

    if not os.path.exists(fig_dir):
        print(f"Error: Directory does not exist - {fig_dir}")
        return

    # Filter pdf_files to ONLY include legend.pdf
    all_files = os.listdir(fig_dir)
    pdf_files = [f for f in all_files if f.lower() == "legend.pdf"]
    
    print(f"Clipping figures in: {fig_dir}")
    if args.inplace:
        print("MODE: In-place (Overwriting original files)\n")
    else:
        print(f"MODE: Creating new files with suffix '{args.suffix}'\n")

    processed_count = 0
    for filename in pdf_files:
        input_file = os.path.join(fig_dir, filename)
        
        # Determine base trims for this file type
        base_trim = None
        for key, val in trims.items():
            if filename.lower().endswith(key.lower()):
                base_trim = val.copy()
                break
        
        if base_trim is None:
            continue
            
        # Determine if any override was provided
        any_override = any(v is not None for v in [args.left, args.bottom, args.right, args.top])
            
        if any_override:
            # If user provided ANY override, we only use their values and default the rest to 0
            l = args.left if args.left is not None else 0.0
            b = args.bottom if args.bottom is not None else 0.0
            r = args.right if args.right is not None else 0.0
            t = args.top if args.top is not None else 0.0
        else:
            # Use standard base trims if no overrides
            l = base_trim["left"]
            b = base_trim["bottom"]
            r = base_trim["right"]
            t = base_trim["top"]
            
        if args.inplace:
            output_file = input_file
        else:
            output_name = filename.replace(".pdf", f"{args.suffix}.pdf")
            output_file = os.path.join(fig_dir, output_name)
        
        clip_pdf(input_file, output_file, l, b, r, t)
        processed_count += 1

    if processed_count == 0:
        print("No 'legend.pdf' found to process.")

if __name__ == "__main__":
    main()
