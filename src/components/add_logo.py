from PIL import Image, ImageDraw, ImageOps

def add_logo(logo_path, width, height, radius):
    logo = Image.open(logo_path)
    logo = logo.resize((width, height))
 
    # Create a mask for rounded corners
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
 
    # Apply the mask to the logo
    rounded_logo = ImageOps.fit(logo, (width, height), centering=(0.5, 0.5))
    rounded_logo.putalpha(mask)
 
    return rounded_logo