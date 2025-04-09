from ais_rag.vectordb import Retriever, FAISSDB
from jinja2 import Template
class PromptBuilder:
    def __init__(self, retriever: Retriever, template_str: str):
        self.retriever = retriever
        self.template = Template(template_str)

    def build(self, query: str) -> str:
        context = self.retriever.context(query)
        return self.template.render(context=context, question=query)
