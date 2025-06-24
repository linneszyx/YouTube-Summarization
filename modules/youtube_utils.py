import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url, language_code='en'):
    video_id = get_video_id(url)
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = ""
    for t in transcripts:
        if t.language_code == language_code:
            try:
                transcript = t.fetch()
                break
            except Exception:
                continue
        elif t.language_code == 'en' and not transcript:
            transcript = t.fetch()
    return transcript if transcript else None

def process(transcript):
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except KeyError:
            pass
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(processed_transcript)
