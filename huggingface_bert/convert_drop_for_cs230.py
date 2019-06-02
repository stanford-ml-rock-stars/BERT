import json
from typing import List, Tuple
## from reading_comprehension.data.drop_reader import DROPReader
## BA outcomment above and new line below:
from drop_reader_for_cs230 import DROPReader                                    ## changed name

## BA insert:
#drop_data_path_train='/Users/ba/Downloads/cs230/project/BERT/DATA/drop_dataset_train.json'
#drop_data_path_dev='/Users/ba/Downloads/cs230/project/BERT/DATA/drop_dataset_dev.json'

def convert(drop_data_path,
            squad_style_output_path,
            skip_invalid,
            use_matched_span_as_answer_text):
    print(f"Converting: {drop_data_path}")
    reader = DROPReader(instance_format='drop',                                 ## changed from "squad" to "drop"
                        skip_when_all_empty=["passage_span"] if skip_invalid else None,
                        relaxed_span_match_for_finding_labels=use_matched_span_as_answer_text)
    instances = reader.read(drop_data_path)
    print(f"Totally {len(list(instances))} instances")

    #instances = instances[:1000]      ## test
    #print(instances)                  ## test
    instance_count = 0
    skipped_instances = 0
    instances_grouped_by_passage = {}
    for instance in instances:
        passage_id = instance.fields["metadata"].metadata["passage_id"]

        if instance.fields["metadata"].metadata["answer_texts"]:
            if passage_id in instances_grouped_by_passage:
                instances_grouped_by_passage[passage_id].append(instance)
            else:
                instances_grouped_by_passage[passage_id] = [instance]
            instance_count += 1
        else: skipped_instances += 1
    print('skipped instances and total instances afterwards: ', skipped_instances, instance_count)  ## added print and counters above

    squad_style_data = []
    for passage_id, instances in instances_grouped_by_passage.items():
        paragraph_text = instances[0].fields["metadata"].metadata["original_passage"]
        qas = []
        for instance in instances:
            metadata = instance.fields["metadata"].metadata

            question_converted_result = convert_span_answers(metadata["question_token_offsets"],
                                                             instance.fields["answer_as_question_spans"],
                                                             metadata["original_question"],
                                                             single_span=True)
            passage_converted_result = convert_span_answers(metadata["passage_token_offsets"],
                                                            instance.fields["answer_as_passage_spans"],
                                                            metadata["original_passage"],
                                                            single_span=True)

            qas.append({"id": metadata["question_id"],
                        "question": metadata["original_question"],
                        "answer_type": metadata["answer_type"],
                        "answers_as_question_spans": question_converted_result,
                        "answers_as_passage_spans": passage_converted_result,
                        "answer_as_add_sub_expressions": [expression.labels for expression in instance.fields["answer_as_add_sub_expressions"]],
                        "answer_as_add_sub_numbers": [token.text for token in instance.fields["answer_as_add_sub_expressions"][0].sequence_field.tokens],
                        "answer_as_counts":[labelfield.label for labelfield in instance.fields["answer_as_counts"]]})                                       ## added
        new_passage = {"title": passage_id,
                       "paragraphs": [{"context": paragraph_text,
                                       "qas": qas}]}
        squad_style_data.append(new_passage)
    print(len(squad_style_data))                                                ## added print
    with open(squad_style_output_path, "w") as fout:
        squad_style_data = {"data": squad_style_data, "version": "drop-1.0"}
        json.dump(squad_style_data, fout)


def convert_span_answers(token_offsets: List[Tuple[int, int]],
                         answer_spans,
                         text: str,
                         single_span: bool):
    answers = []
    for spanfield in answer_spans:

        if spanfield.span_start == -1 and spanfield.span_end -1:
            answer_start = -1
            answer_text = ""
            is_impossible = True                                                ## added
        else:
            answer_start = token_offsets[spanfield.span_start][0]
            answer_end = token_offsets[spanfield.span_end][1]
            answer_text = text[answer_start : answer_end]
            is_impossible = False                                               ## added
        answers.append({
            "answer_start": answer_start,
            "text": answer_text,
            "is_impossible": is_impossible                                      ## added
        })
        if single_span:
            break
    return answers

def main():
    convert("drop_dataset_train.json",                  ## ...
            "drop_train_for_cs230.json",                ## changed name
            skip_invalid=False,                         ## changed to False
            use_matched_span_as_answer_text=False)      ## changed to False

    ## BA outcomment (because already done):
    convert("drop_dataset_dev.json",
            "drop_dev_for_cs230.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)

    ## BA outcomment:
    """convert("drop_test.json",
            "drop_squad_style_test.json",
            skip_invalid=False,
            use_matched_span_as_answer_text=False)
            """

if __name__ == "__main__":
    main()
