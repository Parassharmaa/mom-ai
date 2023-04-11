import openai 
import pydub
import sys
import io
import os
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
    
mp3_files = [chunk_to_mp3(chunk) for chunk in chunks]

full_text = ""

print("Total chunks: ", len(mp3_files))

# transcribe each chunk
def transcribe_chunk(i, mp3):
    random_name = f"whisper-{i}.mp3"
    mp3.name = random_name
    data = openai.Audio.transcribe("whisper-1", mp3)
    print(f"Chunk {i} transcribed.")
    return data["text"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    transcribed_texts = list(executor.map(transcribe_chunk, range(len(mp3_files)), mp3_files))

full_text = "".join(transcribed_texts)

print(full_text)

# save full text to file
file_name = file_key.split("/")[-1].split(".")[0]

with open(f"{file_name}.txt", "w") as f:
    f.write(full_text)
    f.close()