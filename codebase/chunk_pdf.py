import pdfplumber, re

def chunk_pdf(path, max_len=500):
    with pdfplumber.open(path) as pdf:
        text = " ".join(p.extract_text() or "" for p in pdf.pages)

    text = re.sub(r"\s+", " ", text).strip()
    chunks = []

    for i in range(0, len(text), max_len):
        chunks.append(text[i:i+max_len])

    return chunks

if __name__ == "__main__":
    chunks = chunk_pdf("data/14_Paging.pdf")
    print(f"{len(chunks)} chunks generated")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}:")
        print(c)
        print("-" * 80)