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

    except Exception as e:
        print(f"Failed to clip {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clip PDF margins for UAV paper figures.")
    parser.add_argument("-l", "--left", type=float, default=335, help="Points to trim from left (default: 40)")
    parser.add_argument("-b", "--bottom", type=float, default=75, help="Points to trim from bottom (default: 40)")
    parser.add_argument("-r", "--right", type=float, default=405, help="Points to trim from right (default: 40)")
    parser.add_argument("-t", "--top", type=float, default=65, help="Points to trim from top (default: 40)")
    parser.add_argument("--suffix", type=str, default="_clipped", help="Suffix for output files (default: _clipped)")
    
    args = parser.parse_args()

    # Define the directory containing the project figures
    # Using absolute path based on the user's workspace structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "AFSACPSO global results")
    
    targets = [
        "plot_Algorithm_Comparison_Scenario1_3D.pdf",
        "plot_Algorithm_Comparison_Scenario2_3D.pdf",
        "plot_Algorithm_Comparison_Scenario3_3D.pdf"
    ]

    print(f"Clipping figures in: {fig_dir}")
    print(f"Settings: Left={args.left}, Bottom={args.bottom}, Right={args.right}, Top={args.top}\n")

    for filename in targets:
        input_file = os.path.join(fig_dir, filename)
        output_name = filename.replace(".pdf", f"{args.suffix}.pdf")
        output_file = os.path.join(fig_dir, output_name)
        
        clip_pdf(input_file, output_file, args.left, args.bottom, args.right, args.top)

if __name__ == "__main__":
    main()
