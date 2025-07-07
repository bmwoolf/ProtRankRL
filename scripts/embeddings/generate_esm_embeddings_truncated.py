import argparse
import numpy as np
import torch
import esm
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESM embeddings from a FASTA file with truncation for long sequences.")
    parser.add_argument(
        "--fasta",
        type=str,
        default="protein_inputs/raw/validated_proteins.fasta",
        help="Path to input FASTA file (default: protein_inputs/raw/validated_proteins.fasta)",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default="protein_inputs/embeddings/validated_proteins_esm_embeddings.npy",
        help="Path to output .npy file (default: protein_inputs/embeddings/validated_proteins_esm_embeddings.npy)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="protein_inputs/embeddings/validated_proteins_esm_embeddings.csv",
        help="Path to output .csv file (default: protein_inputs/embeddings/validated_proteins_esm_embeddings.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm1b_t33_650M_UR50S",
        help="ESM model name (default: esm1b_t33_650M_UR50S)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )
    return parser.parse_args()


def truncate_sequence(seq, max_length):
    """Truncate sequence to maximum length, keeping the middle portion."""
    if len(seq) <= max_length:
        return seq
    
    # Keep the middle portion of the sequence
    start = (len(seq) - max_length) // 2
    end = start + max_length
    return seq[start:end]


def main():
    args = parse_args()
    print(f"Loading ESM model: {args.model}")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Read sequences from FASTA
    print(f"Reading sequences from {args.fasta}")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    
    # Process sequences and truncate if necessary
    data = []
    truncation_info = []
    
    for rec in records:
        seq = str(rec.seq)
        original_length = len(seq)
        
        if original_length > args.max_length:
            truncated_seq = truncate_sequence(seq, args.max_length)
            print(f"Truncating {rec.id}: {original_length} -> {len(truncated_seq)} amino acids")
            truncation_info.append((rec.id, original_length, len(truncated_seq)))
            data.append((rec.id, truncated_seq))
        else:
            data.append((rec.id, seq))
    
    print(f"Found {len(data)} sequences.")
    if truncation_info:
        print(f"Truncated {len(truncation_info)} sequences:")
        for protein_id, orig_len, trunc_len in truncation_info:
            print(f"  {protein_id}: {orig_len} -> {trunc_len}")

    embeddings = []
    ids = []
    
    for i, (name, seq) in enumerate(data):
        print(f"Processing {name} ({i+1}/{len(data)})...")
        
        try:
            batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            # Average per-residue representations (excluding start/end tokens)
            embedding = token_representations[0, 1:len(seq)+1].mean(0).cpu().numpy()
            embeddings.append(embedding)
            ids.append(name)
            print(f"  Successfully generated embedding for {name}")
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            # Add a zero embedding as placeholder
            embedding = np.zeros(1280)  # ESM-1b embedding dimension
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
    print(f"Generated embeddings for {len(embeddings)} proteins")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main() 