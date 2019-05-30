import json
import logging
import itertools
import string
from typing import Dict, List, Union, Tuple, Any
from collections import Counter, defaultdict
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension.util import make_reading_comprehension_instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS
from allennlp.data.fields import Field, TextField, MetadataField, LabelField, ListField, \
    SequenceLabelField, SpanField, IndexField
from word2number.w2n import word_to_num
## from reading_comprehension.utils import split_tokens_by_hyphen               ## outcomment
from utils_for_bert import split_tokens_by_hyphen                               ## changed name

## BA insert:
# file_path='/Users/ba/Downloads/cs230/project/drop_dataset/drop_dataset_train.json'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("dropbert")
class DROPReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 skip_when_all_empty: List[str] = None,
                 instance_format: str = "drop",
                 relaxed_span_match_for_finding_labels: bool = True) -> None:
        """
        Reads a JSON-formatted DROP dataset file and returns a ``Dataset``

        Parameters
        ----------
        tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
            We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
            Default is ```WordTokenizer()``.
        token_indexers : ``Dict[str, TokenIndexer]``, optional
            We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
            Default is ``{"tokens": SingleIdTokenIndexer()}``.
        lazy : ``bool``, optional (default=False)
            If this is true, ``instances()`` will return an object whose ``__iter__`` method
            reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
        passage_length_limit : ``int``, optional (default=None)
            If specified, we will cut the passage if the length of passage exceeds this limit.
        question_length_limit : ``int``, optional (default=None)
            If specified, we will cut the question if the length of passage exceeds this limit.
        skip_when_all_empty: ``List[str]``, optional (default=None)
            In some cases such as preparing for training examples, you may want to skip some examples
            when there are no gold labels. You can specify on what condition should the examples be
            skipped. Currently, you can put "passage_span", "question_span", "addition_subtraction",
            or "counting" in this list, to tell the reader skip when there are no such label found.
            If not specified, we will keep all the examples.
        instance_format: ``str``, optional (default="drop")
            Since we want to test different kind of models on DROP dataset, and they may require
            different instance format. Current, we support three formats: "drop", "squad" and "bert".
        relaxed_span_match_for_finding_labels : ``bool``, optional (default=True)
            DROP dataset contains multi-span answers, and the date-type answers usually cannot find
            a single passage span to match it, either. In order to use as many examples as possible
            to train the model, we may not want a strict match for such cases when finding the gold
            span labels. If this argument is true, we will treat every span in the multi-span answers
            as correct, and every token in the date answer as correct, too. Note that this will not
            affect evaluation.
        """
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit

        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction", "counting"], \
                f"Unsupported skip type: {item}"

        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        instances, skip_count = [], 0
        for passage_id, passage_info in dataset.items():
            passage_text = passage_info["passage"]
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 passage_tokens)
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1
            # ======================================= REMOVE =====================================================
            if len(instances) > 15:
                break
        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        return instances

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         passage_tokens: List[Token] = None) -> Union[Instance, None]:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)
        # passage_text = question_text
        # passage_tokens = question_tokens
        if self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently, we only have multiple annotations for the dev and test set due to the fact that dev and test sets have "validated answers".
            # We only use the first annotated answer here because we only care about answers in the training set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])

        if not answer_texts:
            # For some reason the all answer fields are empty. We stop processing.
            return None

        # Tokenize the answer text in order to find the matched span based on token
        # Basically the following code splits hypen connected words
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self._tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))

        if self.instance_format == "squad":
            if not answer_texts:
                print(question_id)
                print(passage_id)
            metadata = {}

            # TODO: turn into functions
            # ----------- spans ---------------
            valid_passage_spans = self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            valid_question_spans = self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            if not valid_passage_spans:
                valid_passage_spans.append((-1, -1))
            if not valid_question_spans:
                valid_question_spans.append((-1, -1))

            # the [start, end) index of each token
            passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
            question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

            metadata['passage_token_offsets'] = passage_offsets
            metadata['question_token_offsets'] = question_offsets
            metadata['valid_passage_spans'] = valid_passage_spans
            metadata['valid_question_spans'] = valid_question_spans

            # ----------- counting ---------------
            numbers_in_answer_texts = []
            # Put all the numbers that appear in the answer texts in a list
            # This list is created on ALL types of answers (date, number, spans). It is up to the convert scrip to decide which to use.
            # For number type of answers, the list should only contain one entry.
            for answer_text in answer_texts:
                number = self.convert_word_to_number(answer_text, try_to_include_more_numbers=False)
                if number is not None:
                    numbers_in_answer_texts.append(number)

            # Filter the counting numbers
            # Currently we only support count number 0 ~ 9
            numbers_for_count = list(range(10))
            valid_counts = self.find_valid_counts(numbers_for_count, numbers_in_answer_texts)

            metadata["counting"] = valid_counts

            # ----------- date ---------------
            # TO be continued..
            common_metadata = {
                "answer_type": answer_type,
                "original_answer_texts": answer_texts,
                "original_passage": passage_text,
                "original_question": question_text,
                "passage_id": passage_id,
                "question_id": question_id
            }
            metadata.update(common_metadata)
            fields: Dict[str, Field] = {}
            fields['metadata'] = MetadataField(metadata)
            return Instance(fields)
        # ------------------------- NOT USED BELOW ----------------------------------------------------------
        elif self.instance_format == "bert":
            question_concat_passage_tokens = question_tokens + [Token("[SEP]")] + passage_tokens
            valid_passage_spans = []
            for span in self.find_valid_spans(passage_tokens, tokenized_answer_texts):
                # This span is for `question + [SEP] + passage`.
                valid_passage_spans.append((span[0] + len(question_tokens) + 1,
                                            span[1] + len(question_tokens) + 1))
            if not valid_passage_spans:
                if "passage_span" in self.skip_when_all_empty:
                    return None
                else:
                    valid_passage_spans.append((len(question_concat_passage_tokens) - 1,
                                                len(question_concat_passage_tokens) - 1))
            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans}
            return self.make_bert_drop_instance(question_tokens,
                                                passage_tokens,
                                                question_concat_passage_tokens,
                                                self._token_indexers,
                                                passage_text,
                                                answer_info,
                                                additional_metadata={
                                                        "original_passage": passage_text,
                                                        "original_question": question_text,
                                                        "passage_id": passage_id,
                                                        "question_id": question_id,
                                                        "answer_annotations": answer_annotations})
        elif self.instance_format == "drop":
            numbers_in_passage = []
            number_indices = []
            for token_index, token in enumerate(passage_tokens):
                number = self.convert_word_to_number(token.text)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(token_index)
            # hack to guarantee minimal length of padded number
            numbers_in_passage.append(0)
            number_indices.append(-1)
            numbers_as_tokens = [Token(str(number)) for number in numbers_in_passage]

            valid_passage_spans = \
                self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            valid_question_spans = \
                self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []

            target_numbers = []
            # `answer_texts` is a list of valid answers.
            for answer_text in answer_texts:
                number = self.convert_word_to_number(answer_text)
                if number is not None:
                    target_numbers.append(number)
            valid_signs_for_add_sub_expressions = []
            valid_counts = []
            if answer_type in ["number", "date"]:
                valid_signs_for_add_sub_expressions = \
                    self.find_valid_add_sub_expressions(numbers_in_passage, target_numbers)
            if answer_type in ["number"]:
                # Currently we only support count number 0 ~ 9
                numbers_for_count = list(range(10))
                valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

            type_to_answer_map = {"passage_span": valid_passage_spans,
                                  "question_span": valid_question_spans,
                                  "addition_subtraction": valid_signs_for_add_sub_expressions,
                                  "counting": valid_counts}

            if self.skip_when_all_empty \
                    and not any(type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
                return None

            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
                           "counts": valid_counts}

            return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    numbers_as_tokens,
                                                    number_indices,
                                                    self._token_indexers,
                                                    passage_text,
                                                    answer_info,
                                                    additional_metadata={
                                                            "original_passage": passage_text,
                                                            "original_question": question_text,
                                                            "original_numbers": numbers_in_passage,
                                                            "passage_id": passage_id,
                                                            "question_id": question_id,
                                                            "answer_info": answer_info,
                                                            "answer_annotations": answer_annotations})
        else:
            raise ValueError(f"Expect the instance format to be \"drop\", \"squad\" or \"bert\", "
                             f"but got {self.instance_format}")

    @staticmethod
    def make_marginal_drop_instance(question_tokens: List[Token],
                                    passage_tokens: List[Token],
                                    number_tokens: List[Token],
                                    number_indices: List[int],
                                    token_indexers: Dict[str, TokenIndexer],
                                    passage_text: str,
                                    answer_info: Dict[str, Any] = None,
                                    additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        fields["passage"] = TextField(passage_tokens, token_indexers)
        fields["question"] = TextField(question_tokens, token_indexers)
        number_index_fields = [IndexField(index, fields["passage"]) for index in number_indices]
        fields["number_indices"] = ListField(number_index_fields)
        # This field is actually not required in the model,
        # it is used to create the `answer_as_plus_minus_combinations` field, which is a `SequenceLabelField`.
        # We cannot use `number_indices` field for creating that, because the `ListField` will not be empty
        # when we want to create a new empty field. That will lead to error.
        fields["numbers_in_passage"] = TextField(number_tokens, token_indexers)
        metadata = {"original_passage": passage_text,
                    "passage_token_offsets": passage_offsets,
                    "question_token_offsets": question_offsets,
                    "question_tokens": [token.text for token in question_tokens],
                    "passage_tokens": [token.text for token in passage_tokens],
                    "number_tokens": [token.text for token in number_tokens],
                    "number_indices": number_indices}
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]

            passage_span_fields = \
                [SpanField(span[0], span[1], fields["passage"]) for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, fields["passage"]))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields = \
                [SpanField(span[0], span[1], fields["question"]) for span in answer_info["answer_question_spans"]]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, fields["question"]))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            add_sub_signs_field = []
            for signs_for_one_add_sub_expression in answer_info["signs_for_add_sub_expressions"]:
                add_sub_signs_field.append(
                        SequenceLabelField(signs_for_one_add_sub_expression, fields["numbers_in_passage"]))
            if not add_sub_signs_field:
                add_sub_signs_field.append(
                        SequenceLabelField([0] * len(fields["numbers_in_passage"]), fields["numbers_in_passage"]))
            fields["answer_as_add_sub_expressions"] = ListField(add_sub_signs_field)

            count_fields = [LabelField(count_label, skip_indexing=True) for count_label in answer_info["counts"]]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def make_bert_drop_instance(question_tokens: List[Token],
                                passage_tokens: List[Token],
                                question_concat_passage_tokens: List[Token],
                                token_indexers: Dict[str, TokenIndexer],
                                passage_text: str,
                                answer_info: Dict[str, Any] = None,
                                additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        # This is separate so we can reference it later with a known type.
        fields['passage'] = TextField(passage_tokens, token_indexers)
        fields['question'] = TextField(question_tokens, token_indexers)
        question_and_passage_filed = TextField(question_concat_passage_tokens, token_indexers)
        fields['question_and_passage'] = question_and_passage_filed

        metadata = {'original_passage': passage_text, 'passage_token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }

        if answer_info:
            metadata['answer_texts'] = answer_info["answer_texts"]

            passage_span_fields = \
                [SpanField(span[0], span[1], fields["question_and_passage"])
                 for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, fields["question_and_passage"]))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = date_tokens
            print (answer_texts)
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    """
    for every answer in answer_texts, find **ALL** the matching spans for that answer.
    """
    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = answer_text.lower().split()
            answer_tokens = [token.strip(STRIPPED_CHARACTERS) for token in answer_tokens]
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue

            # Note this will do the matching at every position of the first answer token
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List[int],
                                       targets: List[int],
                                       max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    if eval_value in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int],
                          targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices

    @staticmethod
    def make_spans_instance(question_tokens: List[Token],
                                            passage_tokens: List[Token],
                                            token_indexers: Dict[str, TokenIndexer],
                                            passage_text: str,
                                            token_spans: List[Tuple[int, int]] = None,
                                            answer_texts: List[str] = None,
                                            additional_metadata: Dict[str, Any] = None) -> Instance:
        """
        =================================================================================
        Copied from allennlp/data/dataset_readers/reading_comprehension/util.py
        =================================================================================

        Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
        in a reading comprehension model.

        Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
        ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
        and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
        fields, which are both ``IndexFields``.

        Parameters
        ----------
        question_tokens : ``List[Token]``
            An already-tokenized question.
        passage_tokens : ``List[Token]``
            An already-tokenized passage that contains the answer to the given question.
        token_indexers : ``Dict[str, TokenIndexer]``
            Determines how the question and passage ``TextFields`` will be converted into tensors that
            get input to a model.  See :class:`TokenIndexer`.
        passage_text : ``str``
            The original passage text.  We need this so that we can recover the actual span from the
            original passage that the model predicts as the answer to the question.  This is used in
            official evaluation scripts.
        token_spans : ``List[Tuple[int, int]]``, optional
            Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
            a list because there might be several possible correct answer spans in the passage.
            Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
            annotations on the dev set; this will select the span that the most annotators gave as
            correct).
        answer_texts : ``List[str]``, optional
            All valid answer strings for the given question.  In SQuAD, e.g., the training set has
            exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
            possible answers, which are the aliases for the known correct entity.  This is put into the
            metadata for use with official evaluation scripts, but not used anywhere else.
        additional_metadata : ``Dict[str, Any]``, optional
            The constructed ``metadata`` field will by default contain ``original_passage``,
            ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
            you want any other metadata to be associated with each instance, you can pass that in here.
            This dictionary will get added to the ``metadata`` dictionary we already construct.

        Return:
        Fields: Dict[str, Field]:
            passage: TextField.     [NOT USED]
            question: TextField.    [NOT USED]
            span_start: IndexField. [NOT USED]
            span_end: IndexField.   [NOT USED]
            metadata:
                original_passage: string.
                original_question: string.
                token_offsets: list[tuple[int, int]].
                    start and end CHAR index of each token in the passage. The end index is NOT INCLUDED.
                answer_texts: list[string].
                    answer separated by spans.
                valid_passage_spans: list[tuple[int, int]].
                    start and end WORD index of answer spans in passage. The end index is INCLUDED.
                    There can be multiple spans.
                valid_question_spans: list[tuple[int, int]].
                    start and end WORD index of answer spans in question. The end index is INCLUDED.
                    There can be multiple spans.
                is_impossible: boolean.
                question_tokens: list.     [NOT USED]
                passage_tokens: list.      [NOT USED]
                answer_annotations: list.  [NOT USED]
        """
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}

        # the [start, end) index of each token
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        metadata = {'original_passage': passage_text,
                    'passage_token_offsets': passage_offsets,
                    'question_token_offsets': question_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }
        if answer_texts:
            metadata['answer_texts'] = answer_texts

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)

        # -------------------------------- NOT USED -----------------------------------------------------
        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        fields['passage'] = passage_field
        fields['question'] = TextField(question_tokens, token_indexers)
        if token_spans:
            # There may be multiple answer annotations, so we pick the one that occurs the most.  This
            # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
            # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
            # all of the annotations.  This is why we have a separate official SQuAD metric calculation
            # (the "em" and "f1" metrics use the official script).
            candidate_answers: Counter = Counter()
            for span_start, span_end in token_spans:
                candidate_answers[(span_start, span_end)] += 1
            span_start, span_end = candidate_answers.most_common(1)[0][0]

            fields['span_start'] = IndexField(span_start, passage_field)
            fields['span_end'] = IndexField(span_end, passage_field)
        # ------------------------------------------------------------------------------------------------

        return Instance(fields)

    @staticmethod
    def make_number_instance(question_tokens: List[Token],
                                            passage_tokens: List[Token],
                                            token_indexers: Dict[str, TokenIndexer],
                                            passage_text: str,
                                            token_spans: List[Tuple[int, int]] = None,
                                            answer_texts: List[str] = None,
                                            additional_metadata: Dict[str, Any] = None) -> Instance:
        """
        =================================================================================
        Copied from allennlp/data/dataset_readers/reading_comprehension/util.py
        =================================================================================

        Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
        in a reading comprehension model.

        Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
        ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
        and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
        fields, which are both ``IndexFields``.

        Parameters
        ----------
        question_tokens : ``List[Token]``
            An already-tokenized question.
        passage_tokens : ``List[Token]``
            An already-tokenized passage that contains the answer to the given question.
        token_indexers : ``Dict[str, TokenIndexer]``
            Determines how the question and passage ``TextFields`` will be converted into tensors that
            get input to a model.  See :class:`TokenIndexer`.
        passage_text : ``str``
            The original passage text.  We need this so that we can recover the actual span from the
            original passage that the model predicts as the answer to the question.  This is used in
            official evaluation scripts.
        token_spans : ``List[Tuple[int, int]]``, optional
            Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
            a list because there might be several possible correct answer spans in the passage.
            Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
            annotations on the dev set; this will select the span that the most annotators gave as
            correct).
        answer_texts : ``List[str]``, optional
            All valid answer strings for the given question.  In SQuAD, e.g., the training set has
            exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
            possible answers, which are the aliases for the known correct entity.  This is put into the
            metadata for use with official evaluation scripts, but not used anywhere else.
        additional_metadata : ``Dict[str, Any]``, optional
            The constructed ``metadata`` field will by default contain ``original_passage``,
            ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
            you want any other metadata to be associated with each instance, you can pass that in here.
            This dictionary will get added to the ``metadata`` dictionary we already construct.

        Return:
        Fields: Dict[str, Field]:
            passage: TextField.     [NOT USED]
            question: TextField.    [NOT USED]
            span_start: IndexField. [NOT USED]
            span_end: IndexField.   [NOT USED]
            metadata:
                original_passage: string.
                original_question: string.
                token_offsets: list[tuple[int, int]].
                    start and end CHAR index of each token in the passage. The end index is NOT INCLUDED.
                answer_texts: list[string].
                    answer separated by spans.
                valid_passage_spans: list[tuple[int, int]].
                    start and end WORD index of answer spans in passage. The end index is INCLUDED.
                    There can be multiple spans.
                valid_question_spans: list[tuple[int, int]].
                    start and end WORD index of answer spans in question. The end index is INCLUDED.
                    There can be multiple spans.
                is_impossible: boolean.
                question_tokens: list.     [NOT USED]
                passage_tokens: list.      [NOT USED]
                answer_annotations: list.  [NOT USED]
        """
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        metadata = {'original_passage': passage_text,
                    'token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }
        if answer_texts:
            metadata['answer_texts'] = answer_texts

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)

        # -------------------------------- NOT USED -----------------------------------------------------
        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        fields['passage'] = passage_field
        fields['question'] = TextField(question_tokens, token_indexers)
        if token_spans:
            # There may be multiple answer annotations, so we pick the one that occurs the most.  This
            # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
            # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
            # all of the annotations.  This is why we have a separate official SQuAD metric calculation
            # (the "em" and "f1" metrics use the official script).
            candidate_answers: Counter = Counter()
            for span_start, span_end in token_spans:
                candidate_answers[(span_start, span_end)] += 1
            span_start, span_end = candidate_answers.most_common(1)[0][0]

            fields['span_start'] = IndexField(span_start, passage_field)
            fields['span_end'] = IndexField(span_end, passage_field)
        # ------------------------------------------------------------------------------------------------

        return Instance(fields)
