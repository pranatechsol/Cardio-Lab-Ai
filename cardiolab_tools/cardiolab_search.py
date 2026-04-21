import requests
def cardiolab_search(q,n=5):
    r=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",params={"db":"pubmed","term":q,"retmax":n,"retmode":"json"},timeout=10)
    return ["https://pubmed.ncbi.nlm.nih.gov/"+i for i in r.json()["esearchresult"]["idlist"]]
