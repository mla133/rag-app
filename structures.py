from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    page_content: str
    metadata: Dict
