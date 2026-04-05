from pypdf import PdfReader

def extract_text(file: str) -> list[str]:
    # Text extraction from a PDF
    print('File name')
    print(file)
    reader = PdfReader(file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text())

    return pages