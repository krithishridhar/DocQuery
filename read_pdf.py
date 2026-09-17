import fitz

pdf = fitz.open("sample.pdf")

for page_number, page in enumerate(pdf):
    text = page.get_text()

    print(f"\n--- Page {page_number + 1} ---")
    print(text)