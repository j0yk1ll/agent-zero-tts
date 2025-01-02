import os
import re
import emoji
import torch
import stanza
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Iterator, List, Optional
from kokoro.core import generate
from kokoro.models import build_model
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import struct  # For packing length prefixes
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Xeno TTS API")

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and voicepacks
MODEL = None
VOICEPACKS = {}
AVAILABLE_VOICES = []
STANZA_PIPELINES = {}

# A lock to ensure model usage does not happen concurrently in multiple threads
model_lock = threading.Lock()


class GenerateRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_sky"  # Default voice


def initialize_model_and_voices():
    global MODEL, VOICEPACKS, AVAILABLE_VOICES, STANZA_PIPELINES

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load the model
    MODEL_PATH = os.path.join("kokoro", "v0_19.pth")
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model path {MODEL_PATH} does not exist.")
        raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist.")
    MODEL = build_model(MODEL_PATH, device)
    logging.info("Model loaded successfully.")

    # Load available voices
    VOICEPACKS_DIR = os.path.join("kokoro", "voices")
    if not os.path.isdir(VOICEPACKS_DIR):
        logging.error(f"Voices directory {VOICEPACKS_DIR} does not exist.")
        raise FileNotFoundError(f"Voices directory {VOICEPACKS_DIR} does not exist.")

    for file in os.listdir(VOICEPACKS_DIR):
        if file.endswith(".pt"):
            voice_name = file[:-3]  # Remove '.pt' extension
            VOICEPACK_PATH = os.path.join(VOICEPACKS_DIR, file)
            try:
                voicepack = torch.load(VOICEPACK_PATH, map_location=device)
                VOICEPACKS[voice_name] = voicepack.to(device)
                AVAILABLE_VOICES.append(voice_name)
                logging.info(f"Loaded voice: {voice_name}")
            except Exception as e:
                logging.error(f"Failed to load voice {voice_name}: {e}")

    # Initialize a default Stanza pipeline for English
    LANGUAGE = "en"
    if LANGUAGE not in STANZA_PIPELINES:
        logging.info(f"Initializing Stanza for language: {LANGUAGE}")
        stanza.download(LANGUAGE, processors="tokenize", verbose=False)
        STANZA_PIPELINES[LANGUAGE] = stanza.Pipeline(
            LANGUAGE, processors="tokenize", verbose=False
        )


@app.on_event("startup")
def startup_event():
    initialize_model_and_voices()


def get_stanza_pipeline(language: str):
    """Initialize and/or retrieve the Stanza pipeline for the specified language."""
    if language not in STANZA_PIPELINES:
        logging.info(f"Initializing Stanza for language: {language}")
        stanza.download(language, processors="tokenize", verbose=False)
        STANZA_PIPELINES[language] = stanza.Pipeline(
            language, processors="tokenize", verbose=False
        )
    return STANZA_PIPELINES[language]


def clean_sentence(s: str) -> str:
    s = s.replace('\n', ' ')
    s = s.replace('“', '"').replace('”', '"').replace('’', "'")
    s = re.sub(r'[–—]', '-', s)
    return s.strip()


def generate_sentences(
    text: str,
    language: str = "en",
    cleanup_links: bool = True,
    cleanup_emojis: bool = True,
    max_length: int = 200,  # Maximum number of characters per sentence chunk
) -> Iterator[str]:
    """
    Processes a single text string, cleans it, splits it into well-formed sentences using Stanza,
    further splits long sentences into smaller chunks based on grammatical structure, and yields them one by one.
    """
    nlp = get_stanza_pipeline(language)

    # Clean URLs
    if cleanup_links:
        link_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        text = link_pattern.sub("", text)

    # Clean emojis
    if cleanup_emojis:
        text = emoji.replace_emoji(text, "")

    text = text.strip()
    if not text:
        return

    doc = nlp(text)
    for sentence in doc.sentences:
        s = clean_sentence(sentence.text.strip())

        if s:
            # If the sentence is within the desired length, yield it directly
            if len(s) <= max_length:
                yield s
            else:
                # Further split the long sentence into smaller chunks
                chunks = split_long_sentence(sentence, max_length)
                for chunk in chunks:
                    yield chunk


def split_long_sentence(
    sentence: stanza.models.common.doc.Sentence, max_length: int
) -> Iterator[str]:
    """
    Splits a long sentence into smaller chunks based on punctuation and conjunctions.
    """
    # Potential split points: punctuation marks and coordinating conjunctions
    split_tokens = {",", ";", "and", "but", "or", "so", "yet", "for", "nor"}

    current_chunk = []
    current_length = 0

    for word in sentence.words:
        current_chunk.append(word.text)
        current_length += len(word.text) + 1  # +1 for space or punctuation

        # Check if the current word is a potential split point
        if (
            word.text.lower() in split_tokens or word.deprel == "cc"
        ):  # 'cc' is coordinating conjunction
            if current_length >= max_length:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    yield chunk_text
                current_chunk = []
                current_length = 0

    # Yield any remaining words as the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            yield chunk_text


def encode_audio_to_wav(audio_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """
    Encodes raw audio data to WAV format (16-bit mono).
    """
    import wave

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)
            audio_normalized = np.int16(audio_data * 32767)
            wf.writeframes(audio_normalized.tobytes())
        return buffer.getvalue()


@app.get("/voices", response_model=List[str])
def list_voices():
    """Endpoint to list all available voices."""
    return AVAILABLE_VOICES


@app.post("/generate")
def generate_audio(request: GenerateRequest):
    """
    Endpoint to generate audio from full text.
    Returns a streaming response of multiple WAV chunks (length-prefixed).
    Uses threads to generate each sentence in parallel and streams
    a small “silence WAV” chunk if the next sentence's audio isn’t ready yet,
    thus allowing overlapping TTS and lower latency.

    The ordering of sentences is enforced by:
    1) Locking around the TTS call to avoid mixing up the audio generation.
    2) Streaming in index order (results[i]) to ensure correct sequence.
    """
    text = request.text
    voice = request.voice or "af_sky"

    if voice not in VOICEPACKS:
        raise HTTPException(
            status_code=400, detail=f"Voice '{voice}' is not available."
        )

    voicepack = VOICEPACKS[voice]

    # Convert generator to list so we know how many sentences we have
    sentences_list = list(
        generate_sentences(
            text, language="en", cleanup_links=False, cleanup_emojis=False
        )
    )

    # If no valid sentences, we can return nothing
    if not sentences_list:
        return StreamingResponse(iter([]), media_type="application/octet-stream")

    # Prepare a list to hold the actual WAV chunk (prefix + data) for each sentence
    # None means "not generated yet"
    results = [None] * len(sentences_list)

    def worker(i: int, sentence: str):
        """
        Thread worker function that performs TTS for the given sentence,
        encodes the audio to WAV, then stores it in results[i].
        """
        try:
            logging.info(f"Generating audio for sentence #{i}: {sentence}")
            # Lock around the TTS call to avoid concurrency issues in MODEL usage
            with model_lock:
                audio_chunks, phoneme_chunks = generate(
                    MODEL,
                    sentence,
                    voicepack,
                    lang=voice[0],
                )

            combined_audio = (
                np.concatenate(audio_chunks)
                if audio_chunks
                else np.array([], dtype=np.float32)
            )

            # Normalize volume
            max_amp = np.max(np.abs(combined_audio)) if combined_audio.size else 0
            if max_amp > 0:
                combined_audio = combined_audio / max_amp

            # Encode to WAV
            wav_data = encode_audio_to_wav(combined_audio)

            # Prefix length
            length_prefix = struct.pack(">I", len(wav_data))
            results[i] = length_prefix + wav_data

        except Exception as e:
            logging.error(f"Error generating audio for sentence #{i}: {e}")
            # Insert empty chunk so we don't break the stream
            results[i] = b""

    # Spawn a thread per sentence, so TTS is done (mostly) concurrently
    threads = []
    for i, sentence in enumerate(sentences_list):
        t = threading.Thread(target=worker, args=(i, sentence))
        t.start()
        threads.append(t)

    def audio_generator():
        """
        Yields length-prefixed WAV chunks in the original sentence order.
        If the next sentence's audio isn't ready yet, yields a small silence chunk.
        Once all sentences are done, we finish.
        """
        # Prepare a small 100ms “silence WAV” so we can keep streaming
        # if TTS for the next sentence hasn't finished yet.
        silence_samples = np.zeros(int(0.1 * 24000), dtype=np.float32)  # 100ms
        silence_wav = encode_audio_to_wav(silence_samples)
        silence_prefix = struct.pack(">I", len(silence_wav))
        silence_chunk = silence_prefix + silence_wav

        current_index = 0
        total_sentences = len(sentences_list)

        while current_index < total_sentences:
            # If we have the chunk for the current_index, yield it and move on
            if results[current_index] is not None:
                chunk = results[current_index]
                yield chunk
                current_index += 1
            else:
                # Next sentence not ready yet, so yield 100ms silence
                yield silence_chunk
                # Wait a bit so we don't busy-loop
                time.sleep(0.1)

        # Once we've streamed all sentences, we can join threads to clean up
        for t in threads:
            t.join()

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=generated_audio.wav"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
