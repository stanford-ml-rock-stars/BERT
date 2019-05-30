import json
from typing import Dict, List, Union, Tuple, Any
## from reading_comprehension.data.drop_reader import DROPReader
## BA outcomment above and new line below:
from drop_reader_for_bert import DROPReader                                     ## changed name

## BA insert:
# drop_data_path='/Users/ba/Downloads/cs230/project/drop_dataset/drop_dataset_train.json'

"""
qas:
    id:
    question:
    answers:
        counting: List[int]
        spans:
            question_spans: List
                answer_start
                answer_end
            passage_spans: List
                answer_start
                answer_end
    answer_type:
"""
def convert(drop_data_path,
            squad_style_output_path,
            skip_invalid,
            use_matched_span_as_answer_text):
    print(f"Converting: {drop_data_path}")
    reader = DROPReader(instance_format='squad',
                        skip_when_all_empty=["passage_span"] if skip_invalid else None,
                        relaxed_span_match_for_finding_labels=use_matched_span_as_answer_text)
    instances = reader.read(drop_data_path)
    print(f"Totally {len(list(instances))} instances")

    instance_count = 0
    skipped_instances = 0
    instances_grouped_by_passage = {}
    for instance in instances:
        if instance is None:
            skipped_instances += 1
            continue

        instance_count += 1
        passage_id = instance.fields["metadata"].metadata["passage_id"]

        if passage_id in instances_grouped_by_passage:
            instances_grouped_by_passage[passage_id].append(instance)
        else:
            instances_grouped_by_passage[passage_id] = [instance]

    print('skipped instances and total instances afterwards: ', skipped_instances, instance_count)  ## added print and counters above

    squad_style_data = []
    for passage_id, instances in instances_grouped_by_passage.items():
        paragraph_text = instances[0].fields["metadata"].metadata["original_passage"]
        qas = []
        for instance in instances:
            metadata = instance.fields["metadata"].metadata
            answers = {}

            answer_type = metadata["answer_type"]
            # ---------------- counting --------------------------
            # We only put the count answer if the answer type if "number"
            answers["counting"] = []
            if answer_type == "number":
                # there might be multiple entries for count answer. Therefore the length is not fixed.
                # can be empty if the count is outside 0~9
                answers["counting"] = metadata["counting"]
            if not answers["counting"]:
                answers["counting"] = [-1]

            # ---------------- spans --------------------------
            passage_spans = metadata["valid_passage_spans"]
            question_spans = metadata["valid_question_spans"]
            passage_token_offsets = metadata["passage_token_offsets"]
            question_token_offsets = metadata["question_token_offsets"]

            question_converted_result = convert_span_answers(question_token_offsets, question_spans, single_span=True)
            passage_converted_result = convert_span_answers(passage_token_offsets, passage_spans, single_span=True)

            answers["spans"] = {
                "question_spans": question_converted_result,
                "passage_spans": passage_converted_result
            }


            # ---------------- arithmatics --------------------------
            # TODO:

            # ---------------- date -------------------------
            # TODO:

            qas.append({"id": metadata["question_id"],
                        "question": metadata["original_question"],
                        "answers": answers,
                        "answer_type": answer_type})
        new_passage = {"title": passage_id,
                       "paragraphs": [{"context": paragraph_text,
                                       "qas": qas}]}
        squad_style_data.append(new_passage)
    print('There are', len(squad_style_data), "passages in total")
    with open(squad_style_output_path, "w") as fout:
        squad_style_data = {"data": squad_style_data, "version": "drop-1.0"}
        json.dump(squad_style_data, fout)

def convert_span_answers(token_offsets: List[Tuple[int, int]],
                             answer_spans: List[Tuple[int, int]],
                             single_span: bool):
    answers = []
    for idx, span in enumerate(answer_spans):
        if span == (-1, -1):
            answer_start = -1
            answer_end = -1
        else:
            answer_start = token_offsets[span[0]][0]
            answer_end = token_offsets[span[1]][1]
        answers.append({
            "answer_start": answer_start,
            "answer_end": answer_end
        })
        if single_span:
            break
    return answers

def main():
    convert("../DATA/drop_dataset_train.json",
            "drop_squad_style_train_all_test.json",          ## changed name for all examples
            skip_invalid=False,                         ## changed to False
            use_matched_span_as_answer_text=False)      ## changed to False

    ## BA outcomment (because already done):
    """convert("drop_dev.json",
            "drop_squad_style_dev.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)
            """
    ## BA outcomment:
    """convert("drop_test.json",
            "drop_squad_style_test.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)
            """

if __name__ == "__main__":
    main()
