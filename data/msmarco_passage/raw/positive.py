import random

def load_qrels(qrels_file):
    """
    Load the qrels file.
    :param qrels_file: Path to the qrels file.
    :return: A mapping from query ID to a set of relevant document IDs.
    """
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(docid)
    return qrels

def load_texts(text_file):
    """
    Load the text TSV file.
    :param text_file: Path to the text TSV file.
    :return: A mapping from document ID to text content.
    """
    texts = {}
    with open(text_file, 'r') as f:
        for line in f:
            docid, _, text = line.strip().split('\t', 2)
            texts[docid] = text
    return texts

def generate_new_tsv(query_file, text_file, qrels_file, output_file):
    """
    Generate a new TSV file based on the qrels file.
    :param query_file: Path to the query TSV file.
    :param text_file: Path to the text TSV file.
    :param qrels_file: Path to the qrels file.
    :param output_file: Path to the output TSV file.
    """
    # Load qrels file
    qrels = load_qrels(qrels_file)
    # Load text file
    texts = load_texts(text_file)

    with open(query_file, 'r') as query_f, open(output_file, 'w') as output_f:
        for line in query_f:
            qid, _ = line.strip().split('\t', 1)
            if qid in qrels:
                # Get the set of relevant document IDs for the query
                related_docs = qrels[qid]
                # Randomly select one relevant document ID
                selected_docid = random.choice(list(related_docs))
                if selected_docid in texts:
                    # Get the text content of the selected document
                    text = texts[selected_docid]
                    # Write to the new TSV file
                    output_f.write(f"{qid}\t{text}\n")

query_file = 'train.query.txt'
text_file = 'corpus.tsv'
qrels_file = 'qrels.train.tsv'
output_file = 'positives.tsv'

generate_new_tsv(query_file, text_file, qrels_file, output_file)
