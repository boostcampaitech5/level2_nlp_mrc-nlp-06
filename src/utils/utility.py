import datetime
import re


def make_run_name(admin):
    tz = datetime.timezone(datetime.timedelta(hours=9))
    day_time = datetime.datetime.now(tz=tz)
    run_name = day_time.strftime(f'%m%d_%H:%M:%S_{admin}')

    return run_name

def data_preprocessing(dataset):
    context = []
    answers = []

    for item in dataset:
        def process(text, pat, answer_idx):
            pattern = re.compile(pat)
            for i in re.finditer(pattern, text):
                span = i.span()
                if span[0] < answer_idx:
                    answer_idx -= span[1] - span[0] - 1
            text = re.sub(pattern, " ", text)
            return text, answer_idx
        
        patterns = [r"\n", r"\\n", r"#", r"\s+"]
        i_text = item['context']
        i_answer_idx = item['answers']['answer_start'][0]
        for p in patterns:
            i_text, i_answer_idx = process(i_text, p, i_answer_idx)
        
        context.append(i_text)
        answers.append({"answer_start":[i_answer_idx], "text":[item['answers']['text'][0]]})

    dataset = dataset.remove_columns(column_names='context')
    dataset = dataset.add_column(name='context', column=context)
    dataset = dataset.remove_columns(column_names='answers')
    dataset = dataset.add_column(name='answers', column=answers)
    
    return dataset