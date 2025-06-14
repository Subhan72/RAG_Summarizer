from document_ingestion import load_document, chunk_text
from retrieval import DocumentRetriever
from summarization import Summarizer
import time
import os
import gc
import torch
import re
from nltk.tokenize import sent_tokenize
import nltk

def ensure_nltk_resources():
    """
    Ensure required NLTK resources are downloaded.
    """
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Failed to download NLTK punkt_tab: {str(e)}")

def save_output_to_file(chunks, similarities, summary, input_tokens, output_tokens, latency, output_file):
    """
    Save the retrieved chunks, summary, and metrics to a file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Document Summarization Results ===\n\n")
        f.write("Retrieved Chunks:\n")
        for i, (chunk, similarity) in enumerate(zip(chunks, similarities), 1):
            f.write(f"Chunk {i} (Cosine Similarity: {similarity:.4f}):\n")
            f.write(f"{chunk[:500]}...\n\n")
        f.write("Generated Summary:\n")
        f.write(f"{summary}\n\n")
        f.write("Metrics:\n")
        f.write(f"Input Token Count: {input_tokens}\n")
        f.write(f"Output Token Count: {output_tokens}\n")
        f.write(f"Generation Time: {latency:.2f} seconds\n")

def evaluate_summary(summary, document_text, chunks):
    """
    Evaluate the summary for fluency, coverage, and accuracy.
    Returns a dictionary with scores (0-10) and comments.
    """
    try:
        sentences = sent_tokenize(summary)
        fluency = 8 if len(sentences) >= 2 and all(len(s.split()) > 5 for s in sentences) else 5
        fluency_comments = "Readable with clear sentences." if fluency >= 8 else "Short or fragmented sentences."

        key_terms = ["ai", "generative", "drug discovery", "molecular", "optimization"]
        coverage = sum(1 for term in key_terms if term.lower() in summary.lower()) * 2
        coverage = min(coverage, 10)  # Cap at 10
        coverage_comments = f"Includes {coverage//2} of {len(key_terms)} key terms." if coverage >= 6 else "Misses several key concepts."

        metadata_pattern = re.compile(r'\d{4}, \d+\(\d+\), \d+–\d+|World Journal|Publication history|Corresponding author')
        has_metadata = bool(metadata_pattern.search(summary))
        contradiction = any("not" in summary.lower() and "not" not in chunk.lower() for chunk in chunks)
        accuracy = 7 if not contradiction else 4
        if has_metadata:
            accuracy -= 2
        accuracy_comments = "No contradictions detected." if accuracy >= 7 else "Contains metadata or potential inaccuracies."

        return {
            "fluency": fluency,
            "coverage": coverage,
            "accuracy": accuracy,
            "comments": f"Fluency: {fluency_comments} Coverage: {coverage_comments} Accuracy: {accuracy_comments}"
        }
    except Exception as e:
        return {
            "fluency": 0,
            "coverage": 0,
            "accuracy": 0,
            "comments": f"Evaluation failed: {str(e)}"
        }

def save_evaluation(evaluations, eval_file="evaluation.txt"):
    """
    Save evaluation results for all documents.
    """
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write("=== Evaluation Results ===\n\n")
        for doc_id, eval_data in evaluations.items():
            f.write(f"Document {doc_id}:\n")
            f.write(f"Fluency: {eval_data['fluency']}/10\n")
            f.write(f"Coverage: {eval_data['coverage']}/10\n")
            f.write(f"Accuracy: {eval_data['accuracy']}/10\n")
            f.write(f"Comments: {eval_data['comments']}\n\n")
        f.write("Note: Manual review recommended for accuracy, as automated checks are heuristic-based.")

def main():
    ensure_nltk_resources()
    
    documents = [
        "sample.pdf",  
        "sample2.pdf",  
        "sample3.pdf"   
    ]
    evaluations = {}
    
    
    try:
        retriever = DocumentRetriever()
        summarizer = Summarizer()
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return
    
    for doc_id, file_path in enumerate(documents, 1):
        output_file = f"output_doc{doc_id}.txt"
        print(f"\nProcessing Document {doc_id}: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                print(f"Document {file_path} not found. Skipping.")
                evaluations[doc_id] = {
                    "fluency": 0,
                    "coverage": 0,
                    "accuracy": 0,
                    "comments": f"Document {file_path} not found."
                }
                continue
            
            text = load_document(file_path)
            print("Document loaded successfully.")
            
            chunks = chunk_text(text)
            print(f"Number of chunks: {len(chunks)}")
            
            embeddings = retriever.embed_chunks(chunks)
            retriever.build_index(embeddings)
            
            query = "Summarize the role of generative AI in drug discovery" if doc_id == 1 else "Summarize this document"
            top_chunks, similarities = retriever.retrieve(query, k=5)
            
            print("\nRetrieved Chunks:")
            for i, (chunk, similarity) in enumerate(zip(top_chunks, similarities), 1):
                print(f"Chunk {i} (Cosine Similarity: {similarity:.4f}):")
                print(f"{chunk[:200]}...\n")
            
            print("Generating summary...")
            start_time = time.time()
            summary, input_tokens, output_tokens = summarizer.generate_summary(top_chunks)
            end_time = time.time()
            latency = end_time - start_time
            
            print("Generated Summary:")
            print(summary)
            print("\nMetrics:")
            print(f"Input Token Count: {input_tokens}")
            print(f"Output Token Count: {output_tokens}")
            print(f"Generation Time: {latency:.2f} seconds")
            
            save_output_to_file(top_chunks, similarities, summary, input_tokens, output_tokens, latency, output_file)
            print(f"Results saved to {output_file}")
            
            evaluations[doc_id] = evaluate_summary(summary, text, top_chunks)
            
        except Exception as e:
            print(f"Error processing Document {doc_id}: {str(e)}")
            evaluations[doc_id] = {
                "fluency": 0,
                "coverage": 0,
                "accuracy": 0,
                "comments": f"Failed to process: {str(e)}"
            }
        finally:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    save_evaluation(evaluations)
    print("\nEvaluation results saved to evaluation.txt")

if __name__ == "__main__":
    main()