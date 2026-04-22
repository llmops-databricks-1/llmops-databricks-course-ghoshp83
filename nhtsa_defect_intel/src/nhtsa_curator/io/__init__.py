"""I/O subpackage: NHTSA source readers and the resilient HTTP client.

Each dataset has its own module exposing:

    fetch_<dataset>(...)       -> iterator of dicts (one per logical row)
    download_pdfs(...)         -> writes PDFs to the UC volume (where applicable)

Bronze writers in ``nhtsa_curator.bronze`` consume these iterators.
"""
