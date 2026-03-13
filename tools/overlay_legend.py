from pypdf import PdfReader, PdfWriter, PageObject
import os
import argparse

def overlay_pdf(target_path, legend_path, output_path, tx, ty, scale=1.0):
    """
    Overlays a legend PDF onto a target PDF by first hard-cropping the legend.
    """
    if not os.path.exists(target_path):
        print(f"Error: Target file not found - {target_path}")
        return
    if not os.path.exists(legend_path):
        print(f"Error: Legend file not found - {legend_path}")
        return

    try:
        reader_target = PdfReader(target_path)
        reader_legend = PdfReader(legend_path)
        writer = PdfWriter()

        # Hard-crop the legend content into a clean object
        orig_legend_page = reader_legend.pages[0]
        l_mb = orig_legend_page.mediabox
        l_w = float(l_mb.right) - float(l_mb.left)
        l_h = float(l_mb.top) - float(l_mb.bottom)
        
        # Create a temporary Writer to isolate the legend content
        from pypdf import Transformation
        temp_writer = PdfWriter()
        clean_legend_page = temp_writer.add_blank_page(width=l_w, height=l_h)
        clean_legend_page.merge_transformed_page(orig_legend_page, Transformation().translate(tx=-float(l_mb.left), ty=-float(l_mb.bottom)))
        
        for page in reader_target.pages:
            t_mb = page.mediabox
            # Since clean_legend_page starts at (0,0), we translate to (tx, ty)
            # This tx, ty must be absolute relative to PDF origin, but merge translates relative to (0,0)?
            # merge_transformed_page merges relative to the target's current origin. 
            # If the target is clipped, we must ensure it lands in the right spot.
            
            op = Transformation().scale(sx=scale, sy=scale).translate(tx=tx, ty=ty)
            page.merge_transformed_page(clean_legend_page, op, over=True)
            writer.add_page(page)

        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"Successfully overlaid legend onto: {os.path.basename(target_path)}")

    except Exception as e:
        print(f"Failed to overlay {target_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Overlay a legend PDF onto other PDFs.")
    parser.add_argument("-l", "--legend", type=str, required=True, help="Path to the legend PDF")
    parser.add_argument("-x", "--x-offset", type=float, default=0, help="X offset relative to position")
    parser.add_argument("-y", "--y-offset", type=float, default=0, help="Y offset relative to position")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale for the legend")
    parser.add_argument("-p", "--position", type=str, default="bottom-left", 
                        choices=["bottom-left", "bottom-right", "top-left", "top-right"],
                        help="Position of the legend")
    parser.add_argument("targets", nargs="+", help="Target PDF files to overlay onto")

    args = parser.parse_args()

    # Get legend dimensions once
    reader_legend = PdfReader(args.legend)
    legend_page = reader_legend.pages[0]
    l_mb = legend_page.mediabox
    l_left = float(l_mb.left)
    l_bottom = float(l_mb.bottom)
    l_w = (float(l_mb.right) - l_left) * args.scale
    l_h = (float(l_mb.top) - l_bottom) * args.scale

    for target in args.targets:
        if not os.path.exists(target):
            alt_paths = [
                os.path.join("/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/data/rlbased", os.path.basename(target)),
                os.path.join("/Users/hust-hwashin621m/Desktop/vietanhpaper-2/paper/data/others3chrismast", os.path.basename(target))
            ]
            found = False
            for ap in alt_paths:
                if os.path.exists(ap):
                    target = ap
                    found = True
                    break
            if not found:
                print(f"Skipping (not found): {target}")
                continue

        # Get target dimensions to compute coordinates
        reader_target = PdfReader(target)
        p = reader_target.pages[0]
        mb = p.mediabox
        t_left = float(mb.left)
        t_bottom = float(mb.bottom)
        t_right = float(mb.right)
        t_top = float(mb.top)
        
        # Calculate absolute coordinates (tx, ty) where we want the bottom-left of the legend's visible content to land
        if args.position == "bottom-left":
            tx = t_left + args.x_offset
            ty = t_bottom + args.y_offset
        elif args.position == "bottom-right":
            tx = t_right - l_w - args.x_offset
            ty = t_bottom + args.y_offset
        elif args.position == "top-left":
            tx = t_left + args.x_offset
            ty = t_top - l_h - args.y_offset
        elif args.position == "top-right":
            tx = t_right - l_w - args.x_offset
            ty = t_top - l_h - args.y_offset

        # Call the overlay function with the target absolute coordinates
        # The function internally handles the legend's source offset (l_left, l_bottom).
        overlay_pdf(target, args.legend, target, tx, ty, args.scale)

if __name__ == "__main__":
    main()
