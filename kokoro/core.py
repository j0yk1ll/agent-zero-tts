import phonemizer
import re
import torch
import time

def split_num(num):
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def flip_money(m):
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'

def point_num(num):
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def normalize_text(text):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")  # Replace curly apostrophes with straight apostrophes
    text = text.replace('«', chr(8220)).replace('»', chr(8221))  # Replace guillemets with curly quotes
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')  # Normalize curly quotes to straight quotes
    text = text.replace('(', ',').replace(')', ',')  # Replace "(" and ")" with commas
    text = text.replace(';', '.')  # Replace ";" with "."
    text = text.replace('—', ' ,')  # Replace long dash "—" with " ,"
    text = text.replace('!', '.')  # Replace "!" with "."
    for a, b in zip('、。！，：；？', ',.!,:;?'):  # Replace non-standard punctuation with standard equivalents
        text = text.replace(a, b + ' ')
    text = re.sub(r'[^\S \n]', ' ', text)  # Replace all non-visible characters (except spaces and newlines) with a space
    text = re.sub(r'  +', ' ', text)  # Collapse multiple spaces into a single space
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)  # Remove spaces on empty lines
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)  # Replace "Dr." or "dr." with "Doctor" if followed by a capitalized word
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)  # Replace "Mr." or "MR." with "Mister" if followed by a capitalized word
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)  # Replace "Ms." or "MS." with "Miss" if followed by a capitalized word
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)  # Replace "Mrs." or "MRS." with "Mrs" if followed by a capitalized word
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)  # Replace "etc." with "etc" if not followed by a capitalized word
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)  # Replace "yeah" or "yea" (case-insensitive) with "ye'a"
    text = re.sub(
        r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)',
        split_num,  # Match and process numbers, years, and times
        text
    )
    text = re.sub(r'(?<=\d),(?=\d)', '', text)  # Remove commas inside numbers (e.g., "1,000" -> "1000")
    text = re.sub(
        r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b',
        flip_money,  # Process monetary amounts (e.g., "$10.99")
        text
    )
    text = re.sub(r'\d*\.\d+', point_num, text)  # Process decimal numbers
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)  # Replace hyphens between numbers with " to "
    text = re.sub(r'(?<=\d)S', ' S', text)  # Add a space before "S" following a number
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)  # Normalize possessive "'s" after uppercase letters
    text = re.sub(r"(?<=X')S\b", 's', text)  # Lowercase "S" following "X'" (special case)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)  # Replace repeated initials (e.g., "U.S.") with hyphens (e.g., "U-S")
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)  # Replace periods between uppercase letters with hyphens (e.g., "A.B." -> "A-B")
    return text.strip()

def chunk_text(text, lang, max_tokens=510):
    # Phonemize and tokenize the entire text
    phonemized_text = phonemize(text, lang)
    tokenized_text = tokenize(phonemized_text)

    chunks = []
    start = 0

    while start < len(tokenized_text):
        end = min(start + max_tokens, len(tokenized_text))

        # Look for the nearest punctuation in the current range (prefer closer to `end`)
        best_punctuation_pos = -1
        for pos in range(start, end):
            if phonemized_text[pos] in ".,!?":  # Consider valid punctuation marks
                best_punctuation_pos = pos
                #print(f"Found punctuation at position {pos}: {phonemized_text[pos]}")
        # If we found a punctuation, use it as the chunk's end
        if best_punctuation_pos != -1:
            chunk_end = best_punctuation_pos + 1  # Include the punctuation in the chunk
            #print(f"Using punctuation as chunk end: {chunk_end}")
        else:
            chunk_end = end  # No punctuation found, use the `max_tokens` range
            #print(f"Using max_tokens as chunk end: {chunk_end}")

        # Append the chunk
        chunks.append((
            phonemized_text[start:chunk_end],
            tokenized_text[start:chunk_end]
        ))

        # Move start to the end of the current chunk
        start = chunk_end

    return chunks

def tokens_to_text(tokens):
    """
    Reconstruct text from a list of token IDs using the global VOCAB.
    """
    reverse_dict = {v: k for k, v in VOCAB.items()}
    return ''.join(reverse_dict[t] for t in tokens if t in reverse_dict)

def chunk_text_by_lines(
    text,
    lang,
    max_tokens=510,
    cancellation_flag=None
):
    """
    1) Split the original text by lines (unphonemized).
    2) For each line, phonemize + tokenize.
       - If the line itself has more tokens than max_tokens, break it into forced
         sub-chunks of up to max_tokens.
       - Otherwise, try to accumulate this line into the current chunk if it fits.
         If it doesn't fit, finalize the current chunk and start a new one.
    3) If cancellation_flag is True at any time, we stop generating more chunks immediately.
    4) Return a list of (chunk_text, chunk_tokens) tuples, each guaranteed ≤ max_tokens.
    """

    lines = text.split('\n')
    chunks = []
    current_chunk_tokens = []
    current_chunk_size = 0  # Number of tokens in the current chunk

    for raw_line in lines:
        # Check cancellation before processing each line
        if cancellation_flag and cancellation_flag():
            print("Stopping chunk_text_by_lines due to cancellation_flag.")
            break

        line = raw_line.strip()
        if not line:
            continue  # Skip empty lines

        # 1) Phonemize and tokenize this line
        phonemized_line = phonemize(line, lang)
        line_tokens = tokenize(phonemized_line)

        # 2) If the line itself exceeds max_tokens, break it into forced sub-chunks
        #    before trying to accumulate it in current_chunk_tokens.
        idx = 0
        while idx < len(line_tokens):
            # Check cancellation inside the loop as well
            if cancellation_flag and cancellation_flag():
                print("Stopping chunk_text_by_lines due to cancellation_flag.")
                break

            remaining = len(line_tokens) - idx
            # Sub-chunk = either the whole remainder if it fits in max_tokens,
            # or exactly max_tokens if it's bigger
            sub_chunk_size = min(remaining, max_tokens)

            sub_chunk = line_tokens[idx : idx + sub_chunk_size]
            idx += sub_chunk_size

            # If sub-chunk is the entire (or part of) a large line, we must see if we
            # can accumulate it in the *current* chunk or if we need to finalize first.

            # If sub-chunk alone is bigger than max_tokens (only possible if we forced
            # sub-chunk_size = max_tokens, we’ll still handle partial line in multiple loops).
            # But let's check if we can *accumulate* sub_chunk on top of current_chunk.
            if current_chunk_size + len(sub_chunk) <= max_tokens:
                # We can fit the sub-chunk into the current chunk
                current_chunk_tokens.extend(sub_chunk)
                current_chunk_size += len(sub_chunk)
            else:
                # We cannot fit sub-chunk in the current chunk → finalize the current chunk
                if current_chunk_tokens:
                    chunk_text_reconstructed = tokens_to_text(current_chunk_tokens)
                    chunks.append((chunk_text_reconstructed, current_chunk_tokens))

                # Start a new chunk with the sub-chunk
                current_chunk_tokens = sub_chunk
                current_chunk_size = len(sub_chunk)

            # If the sub-chunk exactly fills the chunk, finalize immediately
            if current_chunk_size == max_tokens:
                chunk_text_reconstructed = tokens_to_text(current_chunk_tokens)
                chunks.append((chunk_text_reconstructed, current_chunk_tokens))
                current_chunk_tokens = []
                current_chunk_size = 0

        # If canceled mid-line chunking, break out of the outer loop
        if cancellation_flag and cancellation_flag():
            print("Stopping chunk_text_by_lines after partial line due to cancellation_flag.")
            break

    # 3) Finalize any leftover tokens in the current chunk if not empty
    if current_chunk_tokens and (not cancellation_flag or not cancellation_flag()):
        chunk_text_reconstructed = tokens_to_text(current_chunk_tokens)
        chunks.append((chunk_text_reconstructed, current_chunk_tokens))

    return chunks


def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()
def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, punctuation_marks=".,!?", with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, punctuation_marks=".,!?", with_stress=True),
)
def phonemize(text, lang, norm=True):
    if norm:
        text = normalize_text(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    
    
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', 'z', ps)
    if lang == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    device = ref_s.device
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()


def generate(
    model,
    text,
    voicepack,
    lang='a',
    speed=1,
    max_tokens=510,
    progress_callback=None,
    cancellation_flag= None
):
    """
    Generate audio from `text` using the provided Kokoro model & voicepack.
    Automatically splits text into token chunks (via `chunk_text_by_lines`).
    Calls `progress_callback(chars_in_chunk, chunk_duration)` after each chunk if provided.
    """
    # 1. Split text into chunks by lines & tokens
    chunks = chunk_text_by_lines(text, lang, max_tokens=max_tokens, cancellation_flag=cancellation_flag)
    audio_output = []
    phonemes_output = []


    # 2. For each chunk, run the TTS
    for i, (chunk_text, chunk_tokens) in enumerate(chunks, 1):
        # Check cancellation right before processing each chunk
        if cancellation_flag and cancellation_flag():
            print("Cancelled before chunk", i)
            break

        print(f"Processing chunk {i} with {len(chunk_tokens)} tokens")
        chunk_start_time = time.time()

        ref_s = voicepack[len(chunk_tokens)]
        out = forward(model, chunk_tokens, ref_s, speed)

        chunk_end_time = time.time()
        chunk_duration = chunk_end_time - chunk_start_time

        if progress_callback:
            progress_callback(len(chunk_text), chunk_duration)

        phonemes_output.append(chunk_text)
        audio_output.append(out)

    return audio_output, phonemes_output


