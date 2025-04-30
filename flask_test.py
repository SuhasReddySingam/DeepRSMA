import requests
url = "http://127.0.0.1:5050/predict"
payload = {
    "rna": "UUUUUUAGGGAAGAUCUGGCCUUCCCACAAGGGAAGGCCAGGGAAUUUUCUU",
    "molecule": "NC1=NC(N)=[NH+]C(NCCCCNC(=[NH2+])C2=CC=C(C=C2)C(=[NH2+])NCCCCNC2=[NH+]C(N)=NC(N)=N2)=N1"
}
print(requests.post(url, json=payload).json())