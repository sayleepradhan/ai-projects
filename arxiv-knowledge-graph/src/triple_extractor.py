import os
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

KG_DELIMITER = "<|>"

EXTRACTION_TEMPLATE = """You are a research knowledge extraction system.
Extract all knowledge triples from the given research paper abstract.
A knowledge triple has the form: (subject, predicate, object)


Rules:
- Subjects and objects should be concise noun phrases (concepts,
  methods, datasets, metrics, or researchers).
- Predicates should be short verb phrases describing the relationship.
- Extract ALL meaningful relationships, including:
  - "X proposes Y", "X outperforms Y", "X is based on Y"
  - "X achieves Y", "X is applied to Y", "X extends Y"
  - Author relationships: "Author proposes Method"
- Separate each triple with the delimiter: {delimiter}
- If no triples can be extracted, output: NONE


EXAMPLE
Abstract: We introduce BERT, a new language representation model.
BERT is designed to pre-train deep bidirectional representations.
It obtains state-of-the-art results on eleven NLP benchmarks.


Output: (BERT, is a, language representation model){delimiter}(BERT,
uses, deep bidirectional representations){delimiter}(BERT, achieves
state-of-the-art on, NLP benchmarks)
END OF EXAMPLE


EXAMPLE
Abstract: Attention mechanisms have become integral to sequence
modeling. The Transformer relies entirely on self-attention,
dispensing with recurrence and convolutions.


Output: (Transformer, relies on, self-attention){delimiter}(Transformer,
dispenses with, recurrence){delimiter}(Transformer, dispenses with,
convolutions){delimiter}(self-attention, is used in, sequence modeling)
END OF EXAMPLE


Now extract triples from this abstract:


Title: {title}
Abstract: {abstract}


Output:"""

def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    """Initialize the Claude LLM."""
    return ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=temperature,
        max_tokens=2048,
    )

def build_extraction_chain(llm=None):
    """Build the LangChain extraction chain."""
    if llm is None:
        llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["title", "abstract", "delimiter"],
        template=EXTRACTION_TEMPLATE,
    )
    return prompt | llm | StrOutputParser()

def extract_triples(
    title: str,
    abstract: str,
    chain=None,
) -> list[tuple[str, str, str]]:
    """Extract knowledge triples from a paper abstract.

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        chain: Optional pre-built chain (for reuse).

    Returns:
        List of (subject, predicate, object) tuples.
    """
    if chain is None:
        chain = build_extraction_chain()

    raw = chain.invoke({
        "title": title,
        "abstract": abstract,
        "delimiter": KG_DELIMITER,
    })
    return parse_triples(raw)


def parse_triples(raw_output: str) -> list[tuple[str, str, str]]:
    """Parse raw LLM output into structured triples.

    Args:
        raw_output: Raw string output from the LLM.

    Returns:
        List of (subject, predicate, object) tuples.
    """
    if not raw_output or raw_output.strip() == "NONE":
        return []

    triples = []
    pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"

    for segment in raw_output.split(KG_DELIMITER):
        segment = segment.strip()
        match = re.search(pattern, segment)
        if match:
            subj = match.group(1).strip()
            pred = match.group(2).strip()
            obj = match.group(3).strip()
            if subj and pred and obj:
                triples.append((subj, pred, obj))

    return triples
