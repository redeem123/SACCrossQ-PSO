import re

def reorder_citations(tex_file):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract citations in order of appearance
    # Regex to capture \cite{key1, key2} or \cite{key1}
    # We need to handle multiple keys in one cite command
    cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
    
    cited_keys_ordered = []
    seen_keys = set()

    for match in cite_pattern.finditer(content):
        # Split by comma and strip whitespace
        keys = [k.strip() for k in match.group(1).split(',')]
        for key in keys:
            if key not in seen_keys:
                cited_keys_ordered.append(key)
                seen_keys.add(key)

    print(f"Found {len(cited_keys_ordered)} unique cited keys.")

    # 2. Extract existing bibitems
    # We need to capture the full bibitem content.
    # A bibitem starts with \bibitem{key} and ends before the next \bibitem or \end{thebibliography}
    
    # First, find the bibliography block
    bib_start_pattern = re.compile(r'\\begin\{thebibliography\}\{99\}')
    bib_end_pattern = re.compile(r'\\end\{thebibliography\}')
    
    bib_start_match = bib_start_pattern.search(content)
    bib_end_match = bib_end_pattern.search(content)
    
    if not bib_start_match or not bib_end_match:
        print("Error: Could not find bibliography environment.")
        return

    bib_content = content[bib_start_match.end():bib_end_match.start()]
    
    # Parse bibitems
    # We'll use a regex that looks ahead for the next bibitem or end of string
    # This is a bit tricky with regex, so we might iterate through matches
    
    bibitem_pattern = re.compile(r'\\bibitem\{([^}]+)\}')
    
    bib_items = {}
    
    # Find all bibitem start positions
    matches = list(bibitem_pattern.finditer(bib_content))
    
    for i, match in enumerate(matches):
        key = match.group(1).strip()
        start_pos = match.start()
        
        if i < len(matches) - 1:
            end_pos = matches[i+1].start()
        else:
            end_pos = len(bib_content)
            
        item_content = bib_content[start_pos:end_pos].strip()
        # Clean up any residual environment commands captured in the item
        item_content = item_content.replace('\\end{sloppypar}', '').replace('\\begin{sloppypar}', '')
        bib_items[key] = item_content

    print(f"Found {len(bib_items)} existing bibitems.")

    # 3. Reconstruct bibliography
    new_bib_content = ""
    
    missing_keys = []
    
    for key in cited_keys_ordered:
        if key in bib_items:
            new_bib_content += bib_items[key] + "\n\n"
        else:
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: The following keys are cited but missing in bibliography: {missing_keys}")
    
    # 4. Replace in original content
    new_full_content = content[:bib_start_match.end()] + "\n" + new_bib_content + content[bib_end_match.start():]
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(new_full_content)
        
    print("Successfully reordered bibliography.")

if __name__ == "__main__":
    reorder_citations(r"c:\Users\PC\Desktop\Copy_of_New Folder - Copy\docs\mypaper\main.tex")
