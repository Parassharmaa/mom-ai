# MOM AI

MOM AI transcribes audio into meeting summary and generate minutes of meeting. Built using Langchain, OpenAI GPT-3, Open Whisper.

## Installation

```bash
poetry install
```

## Usage

#### Generate Audio File Transcription
```bash
poetry run python3 transcribe.py [INSERT AUDIO FILE PATH]
```

#### Generate Meeting Summary
```bash
poetry run python3 generate_summary.py [INSERT TRANSCRIPTION FILE PATH]
```
#### Generate Meeting Minutes
```bash
poetry run python3 generate_mom.py [INSERT TRANSCRIPTION FILE PATH]
```