import openai
import pydub
import sys
import io
import concurrent.futures

# get file key from args
file_key = sys.argv[1]

# divide audio file 10 minutes chunks
audio = pydub.AudioSegment.from_file(file_key)
chunks = pydub.utils.make_chunks(audio, 600000)


# convert chunks to mp3
def chunk_to_mp3(chunk):
    f = io.BytesIO()
    chunk.export(f, format="mp3")
    return f


def mp3_files_gen(chunks, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mp3_file in executor.map(chunk_to_mp3, chunks):
            yield mp3_file


mp3_files = list(mp3_files_gen(chunks))

print(f"Transcribing {len(mp3_files)} chunks...")

full_text = ""


# transcribe each chunk
def transcribe_chunk(i, mp3):
    random_name = f"whisper-{i}.mp3"
    print(f"Transcribing chunk {i} as {random_name}...")
    mp3.name = random_name
    data = openai.Audio.transcribe("whisper-1", mp3)
    print(f"Chunk {i} transcribed.")
    return data["text"]


num_chunks = len(mp3_files)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    transcribed_texts = list(
        executor.map(transcribe_chunk, range(num_chunks), mp3_files)
    )

full_text = "".join(transcribed_texts)

print(full_text)

# save full text to file
file_name = file_key.split("/")[-1].split(".")[0]

with open(f"{file_name}.txt", "w") as f:
    f.write(full_text)
    f.close()
