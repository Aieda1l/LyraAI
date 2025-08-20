# modules/g2p/ollama_g2p.py

import json
import re
import ollama
from rich.console import Console
from rich.panel import Panel

from modules.g2p.base_g2p import BaseG2P

# Initialize a console for beautiful output
CONSOLE = Console()

# This is the system prompt that instructs the LLM on its role and output format.
# It is carefully engineered to ensure reliable JSON output.
SYSTEM_PROMPT = """
You are a linguistic expert specializing in phonetics. Your task is to convert text from various languages into their International Phonetic Alphabet (IPA) representations.

You will be given a JSON object containing "language" and "text".
Your response MUST be a single JSON object. This JSON object should map each word from the input text to a space-separated string of its IPA phonemes.

- Do NOT use markdown code blocks (like ```json).
- Do NOT add any explanations, apologies, or introductory text.
- If a word cannot be transcribed (e.g., punctuation, numbers), map it to an empty string.
- Ensure the keys in the output JSON are the original words from the text.

Example Input:
{"language": "English", "text": "Hello world"}

Example Output:
{"Hello": "həˈloʊ", "world": "wɜːld"}
"""


class OllamaG2P(BaseG2P):
    """
    A Grapheme-to-Phoneme (G2P) converter that uses a local Ollama-served LLM.
    """

    def __init__(self, model: str = 'deepseek-r1', **kwargs):
        """
        Initializes the Ollama G2P converter.

        Args:
            model (str): The name of the model to use via Ollama (e.g., 'deepseek-r1').
        """
        super().__init__(**kwargs)
        self.model = model
        try:
            self.client = ollama.Client()
            # Check if the model is available locally
            self.client.show(self.model)
        except Exception as e:
            CONSOLE.print(Panel(
                f"[bold red]Error initializing Ollama client for model '{self.model}'.[/bold red]\n"
                f"Please ensure Ollama is running and the model is available.\n"
                f"You can pull the model with: [cyan]ollama pull {self.model}[/cyan]\n"
                f"Details: {e}",
                title="Ollama G2P Error",
                border_style="red"
            ))
            raise

    def _g2p(self, input_text: str, lang: str = "English"):
        """
        Performs the G2P conversion by calling the Ollama API.

        Args:
            input_text (str): The text to convert.
            lang (str): The language of the text (e.g., 'English', 'Japanese', 'Korean').

        Returns:
            A tuple of (ph_seq, word_seq, ph_idx_to_word_idx).
        """
        if not input_text.strip():
            return ["SP"], [], [-1]

        # Prepare the payload for the LLM
        user_payload = json.dumps({"language": lang, "text": input_text})

        llm_response_content = ""
        with CONSOLE.status(f"[cyan]Asking '{self.model}' for IPA transcription...", spinner="dots") as status:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_payload}
                    ],
                    options={"temperature": 0.0}  # We want deterministic output
                )
                llm_response_content = response['message']['content']
            except Exception as e:
                CONSOLE.print(f"\n[red]Error calling Ollama API:[/red] {e}")
                return ["SP"], input_text.split(), [-1] * (len(input_text.split()) + 1)

        # --- Parse the LLM response ---
        try:
            # The LLM might wrap the JSON in markdown, so we extract it.
            json_match = re.search(r'\{.*\}', llm_response_content, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No valid JSON object found in the LLM response.", llm_response_content, 0)

            word_to_phoneme_map = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            CONSOLE.print(f"\n[red]Error parsing JSON from LLM response.[/red]")
            CONSOLE.print(f"Details: {e}")
            CONSOLE.print(f"LLM Raw Output:\n---\n{llm_response_content}\n---")
            # Fallback to returning empty phonemes for each word
            words = input_text.split()
            word_to_phoneme_map = {word: "" for word in words}

        # --- Format the output for SOFA ---
        word_seq = list(word_to_phoneme_map.keys())
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]

        for word_idx, word in enumerate(word_seq):
            phonemes_str = word_to_phoneme_map.get(word, "").strip()
            if not phonemes_str:
                # If a word has no phonemes, we skip it but log a warning
                CONSOLE.log(f"[yellow]Warning:[/yellow] No phonemes returned for word: '{word}'")
                continue

            phonemes = phonemes_str.split()
            for ph in phonemes:
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_idx)

            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)

        return ph_seq, word_seq, ph_idx_to_word_idx


# --- For testing the module directly ---
if __name__ == '__main__':
    CONSOLE.print(Panel("[bold magenta]Testing OllamaG2P Module[/bold magenta]"))

    # Make sure Ollama is running with the 'deepseek-r1' model
    try:
        g2p_converter = OllamaG2P(model='deepseek-r1:latest')
    except Exception:
        # Error is already printed in the constructor, so just exit.
        exit(1)

    test_cases = {
        "English": "This is a powerful language model.",
        "Chinese": "街邊太多人與車",  # Traditional Chinese
        "Korean": "나는 가수다",
        "Japanese": "こんにちは世界"
    }

    for lang, text in test_cases.items():
        CONSOLE.print(f"\n--- [bold cyan]Testing Language: {lang}[/bold cyan] ---")
        CONSOLE.print(f"Input Text: {text}")
        ph_seq, word_seq, ph_idx_to_word_idx = g2p_converter._g2p(text, lang=lang)

        print(f"Word Sequence: {word_seq}")
        print(f"Phoneme Sequence: {ph_seq}")
        print(f"Index Map: {ph_idx_to_word_idx}")
        assert len(ph_seq) == len(ph_idx_to_word_idx), "Mismatch between phoneme and index map length!"

    CONSOLE.print("\n[bold green]✅ All tests completed.[/bold green]")