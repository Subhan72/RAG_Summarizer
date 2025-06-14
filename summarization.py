from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import gc

class Summarizer:
    def __init__(self, model_name='facebook/bart-base'):
        """
        Initialize the summarizer with a pre-trained DistilBART model.
        Uses GPU if available, otherwise CPU.
        """
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
            print(f"Model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def generate_summary(self, chunks, max_length=256):
        """
        Generate a summary from the provided list of chunks.
        Returns the summary, input token length, and output token length.
        """
        try:
            gc.collect()
            torch.cuda.empty_cache() if self.device == 'cuda' else None
            
            text = "\n\n".join(chunks)
            inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            output_length = summary_ids.shape[1]
            return summary, input_length, output_length
        except Exception as e:
            raise RuntimeError(f"Summary generation failed: {str(e)}")