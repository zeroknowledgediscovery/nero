import requests
import random
import os
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
import argparse

def fetch_and_extract_papers(num_papers, output_dir, max_results=10000):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch papers directly from arXiv API
    arxiv_url = f"http://export.arxiv.org/api/query?search_query=all&max_results={max_results}"
    response = requests.get(arxiv_url)
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data from arXiv API: {response.status_code}")
    
    # Parse XML response
    root = ET.fromstring(response.text)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    
    if len(entries) < num_papers:
        raise ValueError(f"Not enough papers returned from arXiv ({len(entries)}) to meet requested number ({num_papers}).")
    
    # Randomly select papers
    selected_entries = random.sample(entries, num_papers)
    
    # Process each selected paper
    for entry in selected_entries:
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
        pdf_url = f"http://arxiv.org/pdf/{paper_id}.pdf"
        pdf_path = f"{paper_id}.pdf"

        try:
            # Download the PDF
            print(f"Downloading {pdf_url}...")
            pdf_response = requests.get(pdf_url, stream=True)
            if pdf_response.status_code != 200:
                raise RuntimeError(f"Failed to download PDF for {paper_id}")

            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_response.content)

            # Open the PDF and extract text
            with fitz.open(pdf_path) as pdf_document:
                text = "\n".join(page.get_text() for page in pdf_document)
            # if it is not long enough skip to the next loop
            if(len(text)<100000):
                continue
            # Save the extracted text to a file
            text_filename = os.path.join(output_dir, f"PAPERS_{paper_id}.txt")
            with open(text_filename, "w", encoding="utf-8") as text_file:
                text_file.write(text)

            print(f"Extracted and saved: {text_filename}")

        except Exception as e:
            print(f"Error processing {paper_id}: {e}")

        finally:
            # Clean up the downloaded PDF
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    print(f"Extraction complete. Text files saved in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch, download, and extract text from arXiv papers.")
    parser.add_argument("--num_papers", default=20, type=int, help="Number of papers to download and extract text from")
    parser.add_argument("--output_dir", default="data", type=str, help="Directory to save extracted text files")
    args = parser.parse_args()
    
    fetch_and_extract_papers(args.num_papers, args.output_dir)
