import os
from .record_type import RecordType
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

class Word(RecordType):
    def __init__(self, file_path, content_key='word'):
        self.file_path = file_path
        self.content_key = content_key
    
    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["formatted"] = record.get("formatted")
        return metadata
    
    def init_loader(self):
        return JSONLoader(
            file_path=self.file_path,
            jq_schema='.',  # Each line is a JSON object already
            content_key=self.content_key,
            metadata_func=self.metadata_func,
            text_content=False,
            json_lines=True
        )
