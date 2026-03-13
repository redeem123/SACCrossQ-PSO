import pikepdf
import os
import argparse

def edit_pdf_legend(input_path, output_path, replacements):
    """
    Search and replace text in a PDF's content streams.
    
    Args:
        input_path (str): Path to the input PDF.
        output_path (str): Path to save the modified PDF.
        replacements (dict): Dictionary of {old_text: new_text}.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return

    try:
        with pikepdf.open(input_path) as pdf:
            for page in pdf.pages:
                # Loop through the content streams of the page
                # Some pages have a single stream, some have a list
                contents = page.get('/Contents')
                if contents is None:
                    continue
                
                # Normalize to a list of streams
                if not isinstance(contents, pikepdf.Array):
                    streams = [contents]
                else:
                    streams = contents

                for stream_obj in streams:
                    # Decompress and read the stream
                    data = stream_obj.read_bytes()
                    modified = False
                    
                    for old_text, new_text in replacements.items():
                        # Strings in PDF content streams are often in parentheses (Text)
                        # We try to find the text literally or wrapped in parens
                        # Note: This is a heuristic and might not work for complex encodings
                        targets = [
                            old_text.encode('ascii'),
                            f"({old_text})".encode('ascii')
                        ]
                        
                        for target in targets:
                            if target in data:
                                # If replacing with parens, keep parens but clear content
                                replacement = new_text.encode('ascii') if target == targets[0] else f"({new_text})".encode('ascii')
                                data = data.replace(target, replacement)
                                modified = True
                    
                    if modified:
                        stream_obj.write_bytes(data)

            pdf.save(output_path)
            print(f"Successfully processed: {os.path.basename(input_path)}")
            print(f"  -> Saved to: {output_path}")

    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Edit text in PDF legends (heuristics-based).")
    parser.add_argument("-i", "--input", type=str, help="Input PDF file or directory")
    parser.add_argument("-o", "--output", type=str, help="Output filename or directory")
    parser.add_argument("-s", "--search", type=str, default=" (Global)", help="Text to search for")
    parser.add_argument("-r", "--replace", type=str, default="", help="Text to replace with")
    
    args = parser.parse_args()

    # Defaults for the UAV paper project
    replacements = {args.search: args.replace}
    
    # Identify files to process
    if args.input and os.path.isfile(args.input):
        input_files = [args.input]
    else:
        # Default to the Reward and Convergence plots mentioned by user
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        fig_dir = os.path.join(repo_root, "paper", "figures", "RL_based")
        input_files = [
            os.path.join(fig_dir, "Reward_Comparison_Scenario2.pdf"),
            os.path.join(fig_dir, "Convergence_Comparison_Scenario2.pdf")
        ]

    for input_path in input_files:
        if not os.path.exists(input_path):
            print(f"Skipping (not found): {input_path}")
            continue
            
        # Determine output path
        if args.output:
            if os.path.isdir(args.output):
                output_path = os.path.join(args.output, os.path.basename(input_path))
            else:
                output_path = args.output
        else:
            # Overwrite or create backup
            output_path = input_path.replace(".pdf", "_clean.pdf")
        
        edit_pdf_legend(input_path, output_path, replacements)

if __name__ == "__main__":
    main()
