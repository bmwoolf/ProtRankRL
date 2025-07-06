import argparse
import numpy as np
import torch
import esm
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESM embeddings from a FASTA file.")
    parser.add_argument(
        "--fasta",
        type=str,
        default="protein_inputs/SHRT.fasta",
        help="Path to input FASTA file (default: protein_inputs/SHRT.fasta)",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default="protein_inputs/SHRT_esm_embeddings.npy",
        help="Path to output .npy file (default: protein_inputs/SHRT_esm_embeddings.npy)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="protein_inputs/SHRT_esm_embeddings.csv",
        help="Path to output .csv file (default: protein_inputs/SHRT_esm_embeddings.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm1b_t33_650M_UR50S",
        help="ESM model name (default: esm1b_t33_650M_UR50S)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading ESM model: {args.model}")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Read sequences from FASTA
    print(f"Reading sequences from {args.fasta}")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    data = [(rec.id, str(rec.seq)) for rec in records]
    print(f"Found {len(data)} sequences.")

    embeddings = []
    ids = []
    for i, (name, seq) in enumerate(data):
        print(f"Processing {name} ({i+1}/{len(data)})...")
        batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        # Average per-residue representations (excluding start/end tokens)
        embedding = token_representations[0, 1:len(seq)+1].mean(0).cpu().numpy()
        embeddings.append(embedding)
        ids.append(name)

    embeddings = np.stack(embeddings)
    print(f"Saving embeddings to {args.out_npy} and {args.out_csv}")
    np.save(args.out_npy, embeddings)
    # Save as CSV with IDs
    with open(args.out_csv, "w") as f:
        f.write("id," + ",".join([f"f{i}" for i in range(embeddings.shape[1])]) + "\n")
        for name, emb in zip(ids, embeddings):
            f.write(name + "," + ",".join(map(str, emb)) + "\n")
    print("Done.")


if __name__ == "__main__":
    main() 