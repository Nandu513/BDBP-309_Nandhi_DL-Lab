import numpy as np

SAMPLE_FILE = "sample.csv"
LM_FILE = "map_lm.txt"
TG_FILE = "map_tg.txt"
OUTPUT_FILE = "1000G_float64.npy"

def clean_gene_id(gene_id):
    """Remove version suffix after '.' in gene IDs."""
    return gene_id.split('.')[0]

def load_gene_list(filename):
    """Load and clean Ensembl gene IDs from the *second* column of a mapping file."""
    ids = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                fields = line.strip().split('\t')
                if len(fields) >= 2:  # Ensure second column exists
                    ids.append(clean_gene_id(fields[1]))
    return ids

def find_first_numeric_index(row):
    """Find index of first numeric value in a row."""
    for i, value in enumerate(row):
        try:
            float(value)
            return i
        except ValueError:
            continue
    return None

def safe_float(x):
    """Convert to float, replacing bad values with 0.0."""
    try:
        return float(x)
    except ValueError:
        return 0.0

def main():
    ONE_K_G_genes = []
    RPKM = []

    with open(SAMPLE_FILE) as f:
        header = f.readline().strip().split(",")
        first_data_col = None

        for line in f:
            fields = line.strip().split(",")
            if first_data_col is None:
                first_data_col = find_first_numeric_index(fields)
                if first_data_col is None:
                    raise ValueError("No numeric columns found in the file.")

            gene_id = clean_gene_id(fields[0])  # First col = Ensembl ID
            numeric_values = [safe_float(v) for v in fields[first_data_col:]]

            ONE_K_G_genes.append(gene_id)
            RPKM.append(numeric_values)

    RPKM = np.array(RPKM, dtype=np.float32)

    # Load Ensembl IDs from mapping files
    lm_id = load_gene_list(LM_FILE)
    tg_id = load_gene_list(TG_FILE)

    # Map gene IDs to indices
    gene_to_index = {gene: idx for idx, gene in enumerate(ONE_K_G_genes)}
    lm_idx = [gene_to_index[g] for g in lm_id if g in gene_to_index]
    tg_idx = [gene_to_index[g] for g in tg_id if g in gene_to_index]

    print(f"Total genes in CSV: {len(ONE_K_G_genes)}")
    print(f"Landmark genes matched: {len(lm_idx)}/{len(lm_id)}")
    print(f"Target genes matched: {len(tg_idx)}/{len(tg_id)}")

    if len(lm_idx) < len(lm_id):
        print(f"Warning: {len(lm_id) - len(lm_idx)} landmark genes not found")
    if len(tg_idx) < len(tg_id):
        print(f"Warning: {len(tg_id) - len(tg_idx)} target genes not found")

    if not lm_idx and not tg_idx:
        print("\nExample from CSV IDs:", ONE_K_G_genes[:5])
        print("Example from LM file:", lm_id[:5])
        print("Example from TG file:", tg_id[:5])
        return

    genes_idx = lm_idx + tg_idx
    data = RPKM[genes_idx, :].astype(np.float64)

    print(f"Final matrix shape: {data.shape}")
    print(f"First 5 matched genes: {[ONE_K_G_genes[i] for i in genes_idx[:5]]}")

    np.save(OUTPUT_FILE, data)
    print(f"Saved {data.shape} matrix to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

