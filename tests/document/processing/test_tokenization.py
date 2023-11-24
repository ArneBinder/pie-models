import dataclasses

import pytest
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

from pie_models.document.processing import (
    text_based_document_to_token_based,
    token_based_document_to_text_based,
    tokenize_document,
)
from pie_models.document.processing.tokenization import find_token_offset_mapping
from tests.conftest import TestDocument


@dataclasses.dataclass
class TokenizedTestDocument(TokenBasedDocument):
    sentences: AnnotationList[Span] = annotation_field(target="tokens")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def text_document():
    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.sentences.extend([Span(start=0, end=15), Span(start=16, end=36), Span(start=37, end=54)])
    doc.entities.extend(
        [
            LabeledSpan(start=16, end=24, label="per"),
            LabeledSpan(start=34, end=35, label="org"),
            LabeledSpan(start=41, end=43, label="per"),
            LabeledSpan(start=52, end=53, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_text_document(doc)
    return doc


def _test_text_document(doc):
    assert str(doc.sentences[0]) == "First sentence."
    assert str(doc.sentences[1]) == "Entity M works at N."
    assert str(doc.sentences[2]) == "And it founded O."

    assert str(doc.entities[0]) == "Entity M"
    assert str(doc.entities[1]) == "N"
    assert str(doc.entities[2]) == "it"
    assert str(doc.entities[3]) == "O"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [("Entity M", "per:employee_of", "N"), ("it", "per:founder", "O")]


@pytest.fixture
def token_document():
    doc = TokenizedTestDocument(
        tokens=(
            "[CLS]",
            "First",
            "sentence",
            ".",
            "Entity",
            "M",
            "works",
            "at",
            "N",
            ".",
            "And",
            "it",
            "founded",
            "O",
            ".",
            "[SEP]",
        ),
    )
    doc.sentences.extend(
        [
            Span(start=1, end=4),
            Span(start=4, end=10),
            Span(start=10, end=15),
        ]
    )
    doc.entities.extend(
        [
            LabeledSpan(start=4, end=6, label="per"),
            LabeledSpan(start=8, end=9, label="org"),
            LabeledSpan(start=11, end=12, label="per"),
            LabeledSpan(start=13, end=14, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_token_document(doc)
    return doc


def _test_token_document(doc):
    assert str(doc.sentences[0]) == "('First', 'sentence', '.')"
    assert str(doc.sentences[1]) == "('Entity', 'M', 'works', 'at', 'N', '.')"
    assert str(doc.sentences[2]) == "('And', 'it', 'founded', 'O', '.')"

    assert str(doc.entities[0]) == "('Entity', 'M')"
    assert str(doc.entities[1]) == "('N',)"
    assert str(doc.entities[2]) == "('it',)"
    assert str(doc.entities[3]) == "('O',)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("('Entity', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-cased")


@pytest.fixture(scope="module")
def document_dict(documents):
    return {doc.id: doc for doc in documents}


def test_find_token_offset_mapping(text_document, token_document):
    token_offset_mapping = find_token_offset_mapping(
        text=text_document.text, tokens=list(token_document.tokens)
    )
    assert token_offset_mapping == [
        (0, 0),
        (0, 5),
        (6, 14),
        (14, 15),
        (16, 22),
        (23, 24),
        (25, 30),
        (31, 33),
        (34, 35),
        (35, 36),
        (37, 40),
        (41, 43),
        (44, 51),
        (52, 53),
        (53, 54),
        (54, 54),
    ]


def test_text_based_document_to_token_based(text_document, token_document):
    doc = text_based_document_to_token_based(
        text_document,
        tokens=list(token_document.tokens),
        result_document_type=TokenizedTestDocument,
    )
    _test_token_document(doc)


def test_text_based_document_to_token_based_tokens_from_metadata(text_document, token_document):
    doc = text_document.copy()
    doc.metadata["tokens"] = list(token_document.tokens)
    result = text_based_document_to_token_based(
        doc,
        result_document_type=TokenizedTestDocument,
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_missing_tokens_and_token_offset_mapping(
    text_document, token_document
):
    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            text_document,
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == "tokens or token_offset_mapping must be provided to convert a text based document to "
        "token based, but got None for both"
    )


def test_text_based_document_to_token_based_tokens_from_metadata_are_different(
    text_document, token_document, caplog
):
    doc = text_document.copy()
    doc.metadata["tokens"] = list(token_document.tokens) + ["[PAD]"]
    with caplog.at_level("WARNING"):
        result = text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "tokens in metadata are different from new tokens, take the new tokens"
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_offset_mapping_from_metadata(
    text_document, token_document
):
    doc = text_document.copy()
    doc.metadata["token_offset_mapping"] = find_token_offset_mapping(
        text=doc.text, tokens=list(token_document.tokens)
    )
    result = text_based_document_to_token_based(
        doc,
        result_document_type=TokenizedTestDocument,
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_token_offset_mapping_from_metadata_is_different(
    text_document, token_document, caplog
):
    doc = text_document.copy()
    doc.metadata["token_offset_mapping"] = []
    with caplog.at_level("WARNING"):
        result = text_based_document_to_token_based(
            doc,
            token_offset_mapping=find_token_offset_mapping(
                text=doc.text, tokens=list(token_document.tokens)
            ),
            result_document_type=TokenizedTestDocument,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "token_offset_mapping in metadata is different from the new token_offset_mapping, overwrite the metadata"
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_unaligned_span_strict(text_document, token_document):
    doc = TestDocument(text=text_document.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=0, end=6, label="unaligned"))
    assert str(doc.entities[0]) == "First "

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "First ", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_unaligned_span_not_strict(
    text_document, token_document, caplog
):
    doc = TestDocument(text=text_document.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=0, end=6, label="unaligned"))
    assert str(doc.entities[0]) == "First "

    with caplog.at_level("WARNING"):
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
            strict_span_conversion=False,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == 'cannot find token span for character span "First ", skip it (disable this warning with verbose=False)'
    )

    # check (de-)serialization
    tokenized_doc.copy()

    assert len(doc.entities) == 1
    # the unaligned span is not included in the tokenized document
    assert len(tokenized_doc.entities) == 0


def test_text_based_document_to_token_based_wrong_annotation_type():
    @dataclasses.dataclass
    class WrongAnnotationType(TextBasedDocument):
        wrong_annotations: AnnotationList[Label] = annotation_field(target="text")

    doc = WrongAnnotationType(text="First sentence. Entity M works at N. And it founded O.")
    doc.wrong_annotations.append(Label(label="wrong"))

    with pytest.raises(TypeError) as excinfo:
        text_based_document_to_token_based(
            doc,
            result_document_type=TokenizedTestDocument,
            token_offset_mapping=[],
        )
    assert (
        str(excinfo.value)
        == "can not convert layers that target the text but contain non-span annotations, "
           "but found <class 'pytorch_ie.annotations.Label'> in layer wrong_annotations"
    )


def test_token_based_document_to_text_based(token_document, text_document):
    result = token_based_document_to_text_based(
        token_document,
        text=text_document.text,
        result_document_type=TestDocument,
    )
    _test_text_document(result)


def test_tokenize_document(document_dict, tokenizer):
    doc = document_dict["train_doc2"].copy()
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
    )
    assert len(tokenized_docs) == 1
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == (
        "[CLS]",
        "En",
        "##ti",
        "##ty",
        "A",
        "works",
        "at",
        "B",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
    assert str(doc.sentences[0]) == "Entity A works at B."
    assert (
        str(tokenized_doc.sentences[0]) == "('En', '##ti', '##ty', 'A', 'works', 'at', 'B', '.')"
    )
    assert len(tokenized_doc.entities) == len(doc.entities) == 2
    assert str(doc.entities[0]) == "Entity A"
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
    assert str(doc.entities[1]) == "B"
    assert str(tokenized_doc.entities[1]) == "('B',)"
    assert len(tokenized_doc.relations) == len(doc.relations) == 1
    assert tokenized_doc.relations[0].label == doc.relations[0].label == "per:employee_of"
    assert doc.relations[0].head == doc.entities[0]
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]


def test_tokenize_document_max_length(document_dict, tokenizer):
    doc = document_dict["train_doc2"].copy()
    assert doc.id == "train_doc2"
    assert doc.text == "Entity A works at B."
    assert len(doc.sentences) == 1
    assert str(doc.sentences[0]) == "Entity A works at B."
    assert len(doc.entities) == 2
    assert str(doc.entities[0]) == "Entity A"
    assert str(doc.entities[1]) == "B"
    assert len(doc.relations) == 1
    assert doc.relations[0].label == "per:employee_of"
    assert doc.relations[0].head == doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]

    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        strict_span_conversion=False,
        # This will cut out the second entity. Also, the sentence annotation will be removed,
        # because the sentence is not complete anymore.
        max_length=8,
        return_overflowing_tokens=True,
    )
    assert len(tokenized_docs) == 2
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == ("[CLS]", "En", "##ti", "##ty", "A", "works", "at", "[SEP]")
    assert len(tokenized_doc.sentences) == 0
    assert len(tokenized_doc.entities) == 1
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == ("[CLS]", "B", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 0
    assert len(tokenized_doc.entities) == 1
    assert str(tokenized_doc.entities[0]) == "('B',)"
    assert len(tokenized_doc.relations) == 0


def test_tokenize_document_partition(document_dict, tokenizer):
    doc = document_dict["train_doc7"].copy()
    assert doc.id == "train_doc7"
    assert doc.text == "First sentence. Entity M works at N. And it founded O."
    assert len(doc.sentences) == 3
    assert str(doc.sentences[0]) == "First sentence."
    assert str(doc.sentences[1]) == "Entity M works at N."
    assert str(doc.sentences[2]) == "And it founded O"
    assert len(doc.entities) == 4
    assert str(doc.entities[0]) == "Entity M"
    assert str(doc.entities[1]) == "N"
    assert str(doc.entities[2]) == "it"
    assert str(doc.entities[3]) == "O"
    assert len(doc.relations) == 3
    assert doc.relations[0].head == doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]
    assert doc.relations[1].head == doc.entities[2]
    assert doc.relations[1].tail == doc.entities[3]
    assert doc.relations[2].head == doc.entities[3]
    assert doc.relations[2].tail == doc.entities[2]

    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        strict_span_conversion=False,
        partition_layer="sentences",
    )
    assert len(tokenized_docs) == 3
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc7"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "First", "sentence", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 0
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc7"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "at",
        "N",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 2
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'M')"
    assert str(tokenized_doc.entities[1]) == "('N',)"
    assert len(tokenized_doc.relations) == 1
    assert tokenized_doc.relations[0].label == "per:employee_of"
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]

    tokenized_doc = tokenized_docs[2]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc7"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "And", "it", "founded", "O", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 2
    assert str(tokenized_doc.entities[0]) == "('it',)"
    assert str(tokenized_doc.entities[1]) == "('O',)"
    assert len(tokenized_doc.relations) == 2
    assert tokenized_doc.relations[0].label == "per:founder"
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]
    assert tokenized_doc.relations[1].label == "org:founded_by"
    assert tokenized_doc.relations[1].head == tokenized_doc.entities[1]
    assert tokenized_doc.relations[1].tail == tokenized_doc.entities[0]
