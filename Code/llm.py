from openai import OpenAI
import time
import csv
import re
from tqdm import tqdm


fold = 1
file_name = ""
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
output_csv_name = f"result_test_fold_{fold}_{timestamp}.csv"
start_from = 0

api_key = ""
client = OpenAI(api_key=api_key, base_url="")


def cr_label(code, msg):
    prompt = f"""You are a powerful model in evaluating the clarity of code review comments. Your task is to evaluate the clarity of code review comments based on the following attributes:
    
Relevance: (1) Relevant to the code change. (2) Specify the relevant location. (3) Correctly understand the code change. Mark relevance as "1" if it meets (1) and one of (2) (3), other wise mark as "0"
Informativeness: (1) Clear intention. (2) Provide context information. (3) Provide suggestions for the next step. (4) Provide reference information. Mark informativeness as "1" if it meets (1) (2) and one of (3) (4), other wise mark as "0"
Expression: (1) Concise and to-the-point. (2) Polite and objective. (3) Readable format. (4) Proper syntax and grammar. expression as "1" if it meets (1) (2) and one of (3) (4), other wise mark as "0".
    
Below is the format of an example response:
    
Relevance: 1/0
Informativeness: 1/0
Expression: 1/0
    
Now it's your turn, you only need to output the labels after [output] based on [patch] and [code review comment]. Don't output any additional contents.
[patch]
{code}

[code review comment]
{msg}
       
[output]
"""
    response = client.chat.completions.create(
        model="",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()


def run_api(code, msg):
    retry_limit = 3
    while retry_limit > 0:
        try:
            response = cr_label(code, msg)
            time.sleep(3)
            return response
        except Exception as e:
            print(e)
            retry_limit -= 1
            time.sleep(10)
    return msg


def ignore_non_utf8(string):
    byte_string = string.encode('utf-8', 'ignore')
    clean_string = byte_string.decode('utf-8', 'ignore')
    return clean_string


def extract_mark(attribute, sentence):
    regex_sentence = f'{attribute}: ([10])'
    match = re.search(regex_sentence, sentence)
    if match:
        return match.group(1)
    else:
        print("something wrong happened in extracting the label, use '0' instead")
        return 0


if __name__ == "__main__":
    print(f"Input file name: {file_name}")
    data_column = []
    try:
        with open(file_name, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            if 'patch' in csv_reader.fieldnames:
                data_column = [[row['patch'],row['msg'], row['lang'], row['relevance'], row['informativeness'], row['expression']] for row in csv_reader]
            else:
                print("Error: column not found in the CSV file.")
    except Exception as e:
        print(f"Error: {e}")

    csv_list = []
    with open(output_csv_name, 'w', newline='', encoding ='utf-8') as f:
        print(file_name)
        print(output_csv_name)
        writer = csv.writer(f)
        header = ['patch', 'msg', 'lang', 'relevance_ref', 'informativeness_ref', 'expression_ref', 'relevance_gen', 'informativeness_gen', 'expression_gen']
        if f.tell() == 0:
             writer.writerow(header)
        for data in tqdm(data_column):
            code = ignore_non_utf8(data[0])
            msg = ignore_non_utf8(data[1])
            lang = data[2]
            relevance = data[3]
            info = data[4]
            expression = data[5]
            result = ignore_non_utf8(run_api(code, msg))
            relevance_generated = extract_mark("Relevance", result)
            informativeness_generated = extract_mark("Informativeness", result)
            expression_generated = extract_mark("Expression", result)
            print("--------------------")
            print(f"R Gen: {relevance_generated}")
            print(f"I Gen: {informativeness_generated}")
            print(f"E Gen: {expression_generated}")
            print(f"R Ref: {relevance}")
            print(f"I Ref: {info}")
            print(f"E Ref: {expression}")
            result_list = [code, msg, lang, relevance, info, expression, relevance_generated, informativeness_generated, expression_generated]
            writer.writerow(result_list)
            csv_list.append(result_list)