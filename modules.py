import fitz  # PyMuPDF
import re

def extract_pdf_text():
    doc = fitz.open('pokemon_info.pdf')
    pokemon_info = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pokemon_info.append(text)
    return pokemon_info

# Example usage:
pdf_path = 'pokemon_info.pdf'  # Replace with your PDF path
pokemon_info = extract_pdf_text()
print(pokemon_info)
