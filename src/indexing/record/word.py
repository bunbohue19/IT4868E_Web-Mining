import os
from .record_type import RecordType
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

ROOT = Path(__file__).resolve().parents[3]

WORDS_PATH = f"{ROOT}/dataset/words/[JA-JA]_大辞泉_第二版.jsonl"

class Word(RecordType):
    def __init__(self, content_key='word'):
        self.file_path = WORDS_PATH
        self.content_key = content_key
    
    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["formatted"] = record.get("formatted")
        return metadata
    
    def init_loader(self):
        return JSONLoader(
            file_path=self.file_path,
            jq_schema='.[]',
            content_key=self.content_key,
            metadata_func=self.metadata_func,
            text_content=False,
            json_lines=True
        )