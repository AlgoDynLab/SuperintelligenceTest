import binascii
import copy
import zlib
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

import pandas as pd
from datetime import date
import datetime
from pandas.tseries.offsets import BDay
import torch
from chronos import ChronosPipeline

import math
from nixtla import NixtlaClient
from processing_answers import *
from enchant.utils import levenshtein
from tqdm import tqdm

from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": 192,
        "head_dropout": 0.1,
        "weight_decay": 0,
        "freeze_encoder": True,  # Freeze the patch embedding layer
        "freeze_embedder": True,  # Freeze the transformer encoder
        "freeze_head": False,  # The linear forecasting head must be trained
    },
)
model.init()
print(model)

timegpt = NixtlaClient(
    # defaults to os.environ.get("TIMEGPT_TOKEN")
    # api_key="vGgi9lsf7E4HYwJSnz2m34nAWRO5STxS9hHMJbptSNSmTqMtTylXEEsjNpe8RerEp8Ao02gRzhfJxGi5f2J4OQCgWI6JIIEBMqoTNRNUnrP930rAnmqwwihyLuwgTeo7RYhQSCpJyw5bWVVUddki8l8a1Qh2AH9GITPNlABbOQuX6atPVNSYanYCedXcl8VjbyprosRVhPwFkn0JsN73FKb1VBGGESyaLb8ZMsyGBhUMCfY7M17kkKIrZP2sdc2a"
    api_key="nixak-UjDfWxOwCGaiKbl7tt8YecF4QEiCoKxgzw1DHxW9URqeh5KqFfHW2zRIIbIKS997PLzcI07kChtqEdeP"
)

timegpt.validate_api_key()

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    # "chronos-t5-base",
    device_map="mps",  # "cuda",
    torch_dtype=torch.bfloat16,
)


def check_no_alphabetical_characters_all_string_sequences(list_str_squences):
    counter = 0
    for str_sq in list_str_squences:
        counter = counter + 1
        try:
            str_list = str_sq.split(",")
            num_list = [int(numeric_string) for numeric_string in str_list]
            print(counter, "successfully converted")
        except:
            print(counter, " error ", str_sq)


def ask_pred_by_percentage(one_string_sequence, final_percentage_to_predict):
    str_list = one_string_sequence.split(",")
    num_list = [int(numeric_string) for numeric_string in str_list]
    # print(num_list)
    num_len = len(num_list)
    len_predictions = math.floor((num_len * final_percentage_to_predict) / 100)

    to_be_predicted = num_list[-len_predictions:]
    root = num_list[: num_len - len_predictions]

    # print("num_len: ", num_len)
    # print(len_predictions)
    # print("root: ", root)
    # print("to_be_predicted: ", to_be_predicted)
    return root, to_be_predicted


def ask_pred_by_last_digits(one_string_sequence, num_digits_to_predict):
    str_list = one_string_sequence.split(",")
    num_list = [int(numeric_string) for numeric_string in str_list]
    # print(num_list)
    num_len = len(num_list)

    # Ensure we don't try to predict more digits than exist
    len_predictions = min(num_digits_to_predict, num_len)

    to_be_predicted = num_list[-len_predictions:]
    root = num_list[: num_len - len_predictions]

    # print("num_len: ", num_len)
    # print("len_predictions: ", len_predictions)
    # print("root: ", root)
    # print("to_be_predicted: ", to_be_predicted)
    return root, to_be_predicted


def predict_timeGPT(numerical_context_list, prediction_length):
    """
    Makes predictions using the TimeGPT model based on a numerical sequence.

    Args:
        numerical_context_list (list): List of numbers representing the historical sequence
            to base predictions on.
        prediction_length (int): Number of future values to predict.

    Returns:
        list: List of predicted integer values. Returns empty list if input sequence
            is too short (<=2 values).

    The function:
    1. Converts input numbers to a datetime-indexed dataframe
    2. Uses TimeGPT to forecast future values if sequence is long enough
    3. Rounds predictions to integers
    """
    tiny_df = get_datetime_values(numerical_context_list)

    if len(tiny_df) > 2:
        timegpt_fcst_df = timegpt.forecast(
            tiny_df, h=prediction_length, model="timegpt-1-long-horizon"
        )
        forecasting = list(timegpt_fcst_df["TimeGPT"].values)
        forecasting = [int(round(x)) for x in forecasting]
        return forecasting
    else:
        return []


def predict_Chronos(numerical_context_list, prediction_length):
    """
    Makes predictions using the Chronos model based on a numerical sequence.

    Args:
        numerical_context_list (list): List of numbers representing the historical sequence
            to base predictions on.
        prediction_length (int): Number of future values to predict.

    Returns:
        list: List of predicted integer values rounded from model outputs.

    The function:
    1. Converts input list to PyTorch tensor
    2. Uses pipeline.predict() to generate forecasts with specified parameters
    3. Flattens and processes predictions into a single list of integers
    """
    context = torch.tensor(numerical_context_list)
    forecast = pipeline.predict(
        context=context,
        prediction_length=1,  # <---  number of columns
        num_samples=prediction_length,  # <--- number of predictions
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    forecast = forecast[0].numpy()
    flat_list = [x for xs in forecast for x in xs]
    forecasting = [int(round(x)) for x in flat_list]
    return forecasting


def num_list_to_string(numerical_list):
    return " ".join(str(e) for e in numerical_list)


def add_levenshtein_to_df(one_complexity_df, percentages_list):

    for prctg in percentages_list:
        total_lev_results_list = []

        expected_str_list = one_complexity_df["to_predict_" + str(prctg)].values
        predicted_str_list = one_complexity_df["forecasted_" + str(prctg)].values

        for i in range(len(expected_str_list)):
            lvtn = levenshtein(predicted_str_list[i], expected_str_list[i])
            total_lev_results_list.append(lvtn)

        one_complexity_df["levenshtein_" + str(prctg)] = total_lev_results_list
    return one_complexity_df


def compare_predictions(num_predictions, num_original):

    # sort similarity percentage
    similar_list = []
    for i in range(len(num_predictions)):
        similar_list.append(num_predictions[i] == num_original[i])

    sort_similarity = (
        len([ele for ele in similar_list if ele is True]) / len(similar_list) * 100
        if similar_list
        else 0
    )
    # General similarity percentage
    common_elements = set(num_predictions).intersection(set(num_original))
    num_common_elements = len(common_elements)

    # Find the total number of unique elements in both lists
    total_elements = set(num_predictions).union(set(num_original))
    num_total_elements = len(total_elements)

    # Calculate the percentage similarity
    general_similarity = (num_common_elements / num_total_elements) * 100

    # LEVENSHTEIN
    pred_str = num_list_to_string(num_predictions)
    orig_str = num_list_to_string(num_original)
    # print(pred_str, " ", orig_str)
    lvtn = levenshtein(pred_str, orig_str)
    return sort_similarity, general_similarity, lvtn


def simple_comparison_predictions(num_predictions, num_original):
    similar_list = []
    for i in range(len(num_predictions)):
        similar_list.append(num_predictions[i] == num_original[i])
    return 1 if all(similar_list) else 0


def single_measure_prediction_by_last_digits(
    str_original_sequence, num_digits_to_predict, forecast_algo, verbose=False
):
    num_root, num_to_be_predicted = ask_pred_by_last_digits(
        one_string_sequence=str_original_sequence,
        num_digits_to_predict=num_digits_to_predict,
    )
    if forecast_algo == "timeGPT":
        forecasted = predict_timeGPT(
            numerical_context_list=num_root, prediction_length=len(num_to_be_predicted)
        )
    elif forecast_algo == "chronos":
        forecasted = predict_Chronos(
            numerical_context_list=num_root, prediction_length=len(num_to_be_predicted)
        )
    # return 1 if all(forecasted == num_to_be_predicted) else 0
    success = simple_comparison_predictions(
        num_predictions=forecasted, num_original=num_to_be_predicted
    )
    return (str(num_root), str(num_to_be_predicted), str(forecasted), str(success))


def single_measure_prediction_by_precentage(
    str_original_sequence, percentage_to_predict, forecast_algo, verbose=False
):
    num_root, num_to_be_predicted = ask_pred_by_percentage(
        one_string_sequence=str_original_sequence,
        final_percentage_to_predict=percentage_to_predict,
    )
    expected_predicted_len = len(num_to_be_predicted)

    if forecast_algo == "timeGPT":
        forecasted = predict_timeGPT(
            numerical_context_list=num_root, prediction_length=expected_predicted_len
        )
    elif forecast_algo == "chronos":
        forecasted = predict_Chronos(
            numerical_context_list=num_root, prediction_length=expected_predicted_len
        )

    print("root: ", str(num_root))
    print("to be predicted: ", str(num_to_be_predicted))
    # print("forecasted: ", str(forecasted))

    sort_similarity, general_similarity, lvtn = compare_predictions(
        num_predictions=forecasted, num_original=num_to_be_predicted
    )

    if verbose:
        print("root: ", str(num_root))
        print("to be predicted: ", str(num_to_be_predicted))
        print("forecasted: ", str(forecasted))
        print("sorted_similarity: ", sort_similarity)
        print("general_similarity: ", general_similarity)
        print("levenshtein: ", lvtn)

    return (
        sort_similarity,
        general_similarity,
        lvtn,
        str(num_root),
        str(num_to_be_predicted),
        str(forecasted),
    )


def list_prediction_last_binary(
    str_list_sequences, forecast_algo, num_digits_to_predict=1, verbose=False
):
    root_list = []
    to_be_predicted_list = []
    forecasted_list = []
    success_list = []

    for str_seq in tqdm(str_list_sequences, desc="predicting sequences"):
        root, to_be_predicted, forecasted, success = (
            single_measure_prediction_by_last_digits(
                str_seq, num_digits_to_predict, forecast_algo, verbose
            )
        )
        root_list.append(root)
        to_be_predicted_list.append(to_be_predicted)
        forecasted_list.append(forecasted)
        success_list.append(success)
    return (root_list, to_be_predicted_list, forecasted_list, success_list)


def list_measure_prediction_by_percentage(
    str_list_sequences, percentage_to_predict, forecast_algo, verbose=False
):
    sort_similarity_list = []
    general_similarity_list = []
    levenshtein_list = []
    root_list = []
    to_be_predicted_list = []
    forecasted_list = []
    counter = 0

    for str_seq in tqdm(str_list_sequences, desc="predicting sequences"):
        counter += 1

        # ensures process sequences whose root is long enough
        str_list = str_seq.split(",")
        num_list = [int(numeric_string) for numeric_string in str_list]
        num_len = len(num_list)
        if num_len >= 10:

            if verbose:
                print("Processing {}".format(counter))

            ss, gs, lvens, str_root, str_to_b_prediected, str_forcasted = (
                single_measure_prediction_by_precentage(
                    str_original_sequence=str_seq,
                    percentage_to_predict=percentage_to_predict,
                    forecast_algo=forecast_algo,
                    verbose=verbose,
                )
            )
            sort_similarity_list.append(ss)
            general_similarity_list.append(gs)
            levenshtein_list.append(lvens)
            root_list.append(str_root)
            to_be_predicted_list.append(str_to_b_prediected)
            forecasted_list.append(str_forcasted)

    return (
        sort_similarity_list,
        general_similarity_list,
        levenshtein_list,
        root_list,
        to_be_predicted_list,
        forecasted_list,
    )


def list_predictions_by_list_percentages(
    list_str_sequences,
    complexity,
    list_num_percentages,
    forecasting_method_name="",
    seq_type="",
    verbose=False,
):
    output_df = pd.DataFrame()
    for pctg in tqdm(list_num_percentages, desc="percentages..."):
        ss_l, gs_l, lvens, root_l, tob_predicted_l, forecasted_l = (
            list_measure_prediction_by_percentage(
                str_list_sequences=list_str_sequences,
                percentage_to_predict=pctg,
                forecast_algo=forecasting_method_name,
                verbose=verbose,
            )
        )
        ss_col_name = "sort_simi_percen_" + str(pctg)
        gs_col_name = "gral_simi_percen_" + str(pctg)
        lvens_col_name = "levenshtein_" + str(pctg)
        forcasted_col_name = "forecasted_" + str(pctg)
        root_l_col_name = "root_" + str(pctg)
        to_predict_col_name = "to_predict_" + str(pctg)
        forecasted_col_name = "forecasted_" + str(pctg)
        output_df[ss_col_name] = ss_l
        output_df[gs_col_name] = gs_l
        output_df[lvens_col_name] = lvens
        output_df[forcasted_col_name] = forecasted_l
        output_df[root_l_col_name] = root_l
        output_df[to_predict_col_name] = tob_predicted_l
        output_df[forecasted_col_name] = forecasted_l

    num_rows = len(output_df)
    complexity_list = [complexity] * num_rows
    method_name_list = [forecasting_method_name] * num_rows
    seq_type_list = [seq_type] * num_rows
    output_df["complexity"] = complexity_list
    output_df["forecasting_method_name"] = method_name_list
    output_df["seq_type"] = seq_type_list

    return output_df


def list_predictions_last_binary(
    str_list_sequences, forecast_algo, num_digits_to_predict=1, verbose=False
):
    root_list, to_be_predicted_list, forecasted_list, success_list = (
        list_prediction_last_binary(
            str_list_sequences, forecast_algo, num_digits_to_predict, verbose
        )
    )
    output_df = pd.DataFrame()
    output_df["root"] = root_list
    output_df["to_be_predicted"] = to_be_predicted_list
    output_df["forecasted"] = forecasted_list
    output_df["success"] = success_list
    return output_df


def rem_time(d):
    return date(d.year, d.month, d.day)


def get_datetime_values(array_values):
    how_many = len(array_values)
    TODAY = rem_time(datetime.datetime.today())
    START_DATE = (TODAY - BDay(how_many)).to_pydatetime().date()

    TODAY = TODAY.strftime("%Y-%m-%d")
    START_DATE = START_DATE.strftime("%Y-%m-%d")

    # print("TODAY:", TODAY)
    # print("START_DATE:", START_DATE)

    syn_timestamp_serie = pd.DataFrame(
        {"date": pd.date_range(START_DATE, TODAY, freq="1D")}
    )
    syn_timestamp_serie = syn_timestamp_serie.head(how_many)
    syn_timestamp_serie["y"] = array_values
    syn_timestamp_serie["ds"] = pd.to_datetime(syn_timestamp_serie["date"])
    syn_timestamp_serie["unique_id"] = "id1"
    syn_timestamp_serie = syn_timestamp_serie[["unique_id", "ds", "y"]]
    return syn_timestamp_serie


def concat_all_results(verbose=False):
    """Load all datasets

    Args:
        verbose (bool, optional): if you wanna see some details. Defaults to False.

    Returns:
        df: a dataframe with all datasets concatenated
    """
    mathe_df = pd.read_csv("data/automatic_mathematica.csv")
    matlab_df = pd.read_csv("data/automatic_matlab.csv")
    py_df = pd.read_csv("data/automatic_python_no_comments.csv")
    arnoldC_df = pd.read_csv("data/automatic_arnoldC.csv")
    js_df = pd.read_csv(
        "data/automatic_javascript.csv"
    )  # this was not labeled with print_code automaticallyF
    cpp_df = pd.read_csv("data/automatic_Cpp.csv")
    r_df = pd.read_csv("data/automatic_R.csv")

    total_df = pd.concat([mathe_df, matlab_df, py_df, arnoldC_df, js_df, cpp_df, r_df])
    if verbose:
        print("Length: ", str(len(total_df)))
        print("Columns: ", list(total_df.columns))
        print("Languages: ", list(total_df["language"].unique()))

    return total_df


def concat_all_compressed_files(verbose=False):
    """Load all datasets

    Args:
        verbose (bool, optional): if you wanna see some details. Defaults to False.

    Returns:
        df: a dataframe with all datasets concatenated
    """
    mathe_df = pd.read_csv("normalized_compressed_Mathematica.csv")
    matlab_df = pd.read_csv("normalized_compressed_Matlab.csv")
    py_df = pd.read_csv("normalized_compressed_Python.csv")
    arnoldC_df = pd.read_csv("normalized_compressed_ArnoldC.csv")
    js_df = pd.read_csv(
        "normalized_compressed_JavaScript.csv"
    )  # this was not labeled with print_code automaticallyF
    cpp_df = pd.read_csv("normalized_compressed_C++.csv")
    r_df = pd.read_csv("normalized_compressed_R.csv")

    total_df = pd.concat([mathe_df, matlab_df, py_df, arnoldC_df, js_df, cpp_df, r_df])
    if verbose:
        print("Length: ", str(len(total_df)))
        print("Columns: ", list(total_df.columns))
        print("Languages: ", list(total_df["language"].unique()))

    return total_df


def str2hex(s):
    return binascii.hexlify(s)


def hex2str(h):
    return binascii.unhexlify(h)


def no_compression_units(code_string, target_seq_str):
    numbers_list = target_seq_str.split(",")
    for i in range(len(numbers_list)):
        numbers_list[i] = numbers_list[i].replace(" ", "")
    # print(numbers_list)
    no_compression_measure = 0
    for n in numbers_list:
        if n in code_string:
            no_compression_measure += 1
    no_compression_percentage = (no_compression_measure * 100) / len(numbers_list)
    return no_compression_percentage


def compress_line(line):
    """returns a line coded

    Args:
        line (str): text to be coded

    Returns:
        str: coded string
    """
    line_encoded_utf8 = line.encode("utf-8")
    # print(f"{line_encoded_utf8=}")

    line_compressed = zlib.compress(line_encoded_utf8)
    # print(f"{line_compressed=}")

    compressed_hex = str2hex(line_compressed)
    # print(f"{compressed_hex=}")

    compressed_hex_str = compressed_hex.decode("utf-8")
    # print(f"{compressed_hex_str=}")
    return compressed_hex_str


def compress_answers(one_language_df):
    """Given a dataset of a single language, compress/code answers and measure its length
    if the coded one

    Args:
        one_language_df (df): target dataframe

    Returns:
        df: a dataframe agmented with the compressed answers and its length
    """
    # compress the original answers
    compressed_df = copy.deepcopy(one_language_df)
    cols = list(compressed_df.columns)
    compressed_df["compressed_answer"] = compressed_df["GPT4_answer"].apply(
        compress_line
    )
    compressed_df["compressed_len"] = compressed_df["compressed_answer"].apply(len)

    # if dataframe is normalized, also compress the normalized answers
    if "normalized_answer" in cols:
        compressed_df["compressed_norm_answer"] = compressed_df[
            "normalized_answer"
        ].apply(compress_line)
        compressed_df["compressed_norm_len"] = compressed_df[
            "compressed_norm_answer"
        ].apply(len)

    # compress original sequences with no code given by chatGPT
    compressed_df["compressed_sequence"] = compressed_df["sequence"].apply(
        compress_line
    )
    compressed_df["compressed_sequence_len"] = compressed_df[
        "compressed_sequence"
    ].apply(len)
    return compressed_df


def normalize_prints_and_no_compression_measure(language_df):
    """Normalize answers taking definiton out the original sequence from the code
    of the answer

    Args:
        language_df (df): a language dataframe

    Returns:
        df: dataframe with normalized answers
    """
    copy_df = copy.deepcopy(language_df)

    sequences = list(copy_df["sequence"].values)
    original_answers = list(copy_df["GPT4_answer"].values)
    original_lengths = list(copy_df["a_length"].values)
    print_cases = list(copy_df["print_code"].values)

    normalized_answer = copy.deepcopy(original_answers)
    normalized_length = copy.deepcopy(original_lengths)
    no_compression_percentage_list = [0] * len(original_answers)

    for i in range(len(print_cases)):
        is_print = print_cases[i] == 1
        if is_print:
            seq = sequences[i]
            n_answer = original_answers[i].replace(seq, "")
            n_length = len(n_answer)
            normalized_length[i] = n_length
            normalized_answer[i] = n_answer

    copy_df["normalized_answer"] = normalized_answer
    copy_df["normalized_length"] = normalized_length

    # compute no compression percentage
    for i in range(len(sequences)):
        no_compression_percentage = no_compression_units(
            original_answers[i], sequences[i]
        )
        no_compression_percentage_list[i] = no_compression_percentage

    copy_df["no_compression_percentage"] = no_compression_percentage_list
    return copy_df


def compute_mean_column(df, column_name):
    return df.loc[:, column_name].mean()


def describe_df(one_language_df, complexity, verbose=False):

    one_complexity_df = copy.deepcopy(
        one_language_df[one_language_df["complexity"] == complexity]
    )

    # print("     applying complexity filtering: ")
    # print("     looking for complexity: ", str(complexity))
    # print(
    #     "      after appply filter: ",
    #     str(list(one_complexity_df["complexity"].unique())),
    # )
    cols = list(one_complexity_df.columns)
    avg_answer_len = None  # original answer length
    norm_avg_answer_len = None  # normalized answer length
    compressed_avg_len = None  # original compressed answer length
    compressed_norm_avg_len = None  # normalied compressed answer length
    compressed_seq_avg_len = None  # pure compressed seq
    avg_no_compression_percentage = None

    # if dataframe has been normalized and compressed
    if ("normalized_answer" in cols) and ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        norm_avg_answer_len = one_complexity_df.loc[:, "normalized_length"].mean()
        compressed_avg_len = one_complexity_df.loc[:, "compressed_len"].mean()
        compressed_norm_avg_len = one_complexity_df.loc[:, "compressed_norm_len"].mean()
        compressed_seq_avg_len = one_complexity_df.loc[
            :, "compressed_sequence_len"
        ].mean()
        avg_no_compression_percentage = one_complexity_df.loc[
            :, "no_compression_percentage"
        ].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length, compress_avg_len] for complexity {}:[{}, {}, {}, {}, {}]".format(
                    complexity,
                    "%.2f" % (norm_avg_answer_len),
                    "%.2f" % (avg_answer_len),
                    "%.2f" % (compressed_avg_len),
                    "%.2f" % (compressed_norm_avg_len),
                    "%.2f" % (compressed_seq_avg_len),
                )
            )

    elif not ("normalized_answer" in cols) and not ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        if verbose:
            print(
                "avg_length for complexity {}: {}".format(
                    complexity, "%.2f" % (avg_answer_len)
                )
            )

    elif ("normalized_answer" in cols) and not ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        norm_avg_answer_len = one_complexity_df.loc[:, "normalized_length"].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length] for complexity {}: [{}, {}]".format(
                    complexity,
                    "%.2f" % (norm_avg_answer_len),
                    "%.2f" % (avg_answer_len),
                )
            )

    elif not ("normalized_answer" in cols) and ("compressed_answer" in cols):
        avg_answer_len = one_complexity_df.loc[:, "a_length"].mean()
        compressed_avg_len = one_complexity_df.loc[:, "compressed_len"].mean()
        compressed_seq_avg_len = one_complexity_df.loc[
            :, "compressed_sequence_len"
        ].mean()

        if verbose:
            print(
                "[norm_avg_lenght, avg_length, compress_avg_len] for complexity {}:[{}, {}, {}]".format(
                    complexity,
                    "%.2f" % (avg_answer_len),
                    "%.2f" % (compressed_avg_len),
                    "%.2f" % (compressed_seq_avg_len),
                )
            )

    return (
        avg_answer_len,  # original length
        norm_avg_answer_len,  # normalized length
        compressed_avg_len,  # compressed original legth
        compressed_norm_avg_len,  # compressed normalized length
        compressed_seq_avg_len,  # compressed pure sequences length
        avg_no_compression_percentage,  # avg no compression percentage
    )


def summarize_df_by_filter(
    one_language_df,
    complexity,
    only_print=False,
    only_correct_exe=False,
    correct_exe_filter=False,
    print_code_filter=False,
    verbose=False,
):
    aux = copy.deepcopy(one_language_df)
    len_before_filter = len(aux)
    # no filter case
    if (
        (correct_exe_filter is False)
        and (print_code_filter is False)
        and (only_print is False)
        and (only_correct_exe is False)
    ):
        language = aux["language"].unique()[0]
        if language == "ArnoldC":
            filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
        else:
            filtered = copy.deepcopy(aux)
    # only print filter applied
    elif only_print is True:
        filtered = copy.deepcopy(aux[aux["print_code"] == 1])
    # only correct filter applied
    elif only_correct_exe is True:
        filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
    # correct that are not prints
    elif (correct_exe_filter is True) and (print_code_filter is False):
        filtered = copy.deepcopy(
            aux[(aux["correct_execution"] == True) & (aux["print_code"] == 0)]
        )
    # incorrect prints
    elif (correct_exe_filter is False) and (print_code_filter is True):
        filtered = copy.deepcopy(
            aux[(aux["correct_execution"] == False) & (aux["print_code"] == 1)]
        )
    # print and correct filters applied
    elif (correct_exe_filter is True) and (print_code_filter is True):
        filtered = copy.deepcopy(aux[aux["correct_execution"] == True])
        filtered = copy.deepcopy(filtered[filtered["print_code"] == 1])

    # print("*** Applying FILTERS ***")
    # print("LOOKING FOR complexity: ", str(complexity))
    # print("filters [c, p]:", str(correct_exe_filter), str(print_code_filter))
    # print("contained complexity:", str(list(filtered["complexity"].unique())))
    len_after_filter = len(filtered)
    if len_after_filter == 0:
        len_after_filter = 1
    coincidence_percentage = (len_after_filter * 100) / len_before_filter

    (
        avg_answer_len,
        norm_avg_answer_len,
        compressed_avg_len,
        compressed_norm_avg_len,
        compressed_seq_avg_len,
        avg_no_compression_percentage,
    ) = describe_df(one_language_df=filtered, complexity=complexity, verbose=verbose)

    return (
        coincidence_percentage,
        avg_answer_len,
        norm_avg_answer_len,
        compressed_avg_len,
        compressed_norm_avg_len,
        compressed_seq_avg_len,
        avg_no_compression_percentage,
        filtered,
    )


def range1(start, end):
    return range(start, end + 1)


def create_language_summary_df(rough_language_df, verbose=False, save_language_name=""):
    normalize_df = normalize_prints_and_no_compression_measure(rough_language_df)
    normalize_df = compress_answers(normalize_df)
    if save_language_name != "":
        normalize_df.to_csv(
            "normalized_compressed_{}.csv".format(save_language_name), index=False
        )

    # Original (nf = no filter)
    nf_percentage_arr = [0] * 4
    nf_avg_len_arr = [0] * 4
    nf_norm_avg_len_arr = [0] * 4
    nf_compr_avg_len_arr = [0] * 4
    nf_compr_norm_avg_len_arr = [0] * 4
    nf_compr_seq_avg_len_arr = [0] * 4
    nf_avg_no_compression_percentage_arr = [0] * 4
    # only print filter
    p_percentage_arr = [0] * 4
    p_avg_len_arr = [0] * 4
    p_norm_avg_len_arr = [0] * 4
    p_compr_avg_len_arr = [0] * 4
    p_compr_norm_avg_len_arr = [0] * 4
    p_compr_seq_avg_len_arr = [0] * 4
    p_avg_no_compression_percentage_arr = [0] * 4
    # only correct filter
    c_percentage_arr = [0] * 4
    c_avg_len_arr = [0] * 4
    c_norm_avg_len_arr = [0] * 4
    c_compr_avg_len_arr = [0] * 4
    c_compr_norm_avg_len_arr = [0] * 4
    c_compr_seq_avg_len_arr = [0] * 4
    c_avg_no_compression_percentage_arr = [0] * 4
    # correct no prints
    cnp_percentage_arr = [0] * 4
    cnp_avg_len_arr = [0] * 4
    cnp_norm_avg_len_arr = [0] * 4
    cnp_compr_avg_len_arr = [0] * 4
    cnp_compr_norm_avg_len_arr = [0] * 4
    cnp_compr_seq_avg_len_arr = [0] * 4
    cnp_avg_no_compression_percentage_arr = [0] * 4
    # incorrect prints
    ip_percentage_arr = [0] * 4
    ip_avg_len_arr = [0] * 4
    ip_norm_avg_len_arr = [0] * 4
    ip_compr_avg_len_arr = [0] * 4
    ip_compr_norm_avg_len_arr = [0] * 4
    ip_compr_seq_avg_len_arr = [0] * 4
    ip_avg_no_compression_percentage_arr = [0] * 4
    # print and correct filter
    p_c_percentage_arr = [0] * 4
    p_c_avg_len_arr = [0] * 4
    p_c_norm_avg_len_arr = [0] * 4
    p_c_compr_avg_len_arr = [0] * 4
    p_c_compr_norm_avg_len_arr = [0] * 4
    p_c_compr_seq_avg_len_arr = [0] * 4
    p_c_avg_no_compression_percentage_arr = [0] * 4

    lan = rough_language_df["language"].unique()[0]
    language_col = [lan, lan, lan, lan]
    complexity_arr = []
    output = pd.DataFrame()

    counter = -1
    for complexity in range1(1, 4):
        counter = counter + 1
        # No filter
        (
            nf_percentage,
            nf_avg_answer_len,
            nf_norm_avg_answer_len,
            nf_compressed_avg_len,
            nf_compressed_norm_avg_len,
            nf_compressed_seq_avg_len,
            nf_no_copression_percentage,
            nf_filtered_df,
        ) = summarize_df_by_filter(normalize_df, complexity, verbose=verbose)

        # only print ("p" prefix) filter applied
        (
            p_percentage,
            p_avg_answer_len,
            p_norm_avg_answer_len,
            p_compressed_avg_len,
            p_compressed_norm_avg_len,
            p_compressed_seq_avg_len,
            p_no_copression_percentage,
            p_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df, complexity, only_print=True, verbose=verbose
        )

        # only correct ("c" prefix) filter applied
        (
            c_percentage,
            c_avg_answer_len,
            c_norm_avg_answer_len,
            c_compressed_avg_len,
            c_compressed_norm_avg_len,
            c_compressed_seq_avg_len,
            c_no_copression_percentage,
            c_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df, complexity, only_correct_exe=True, verbose=verbose
        )

        # Correct that are not print
        (
            cnp_percentage,
            cnp_avg_answer_len,
            cnp_norm_avg_answer_len,
            cnp_compressed_avg_len,
            cnp_compressed_norm_avg_len,
            cnp_compressed_seq_avg_len,
            cnp_no_copression_percentage,
            cnp_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            correct_exe_filter=True,
            print_code_filter=False,
            verbose=verbose,
        )

        # incorrect prints
        (
            ip_percentage,
            ip_avg_answer_len,
            ip_norm_avg_answer_len,
            ip_compressed_avg_len,
            ip_compressed_norm_avg_len,
            ip_compressed_seq_avg_len,
            ip_no_copression_percentage,
            ip_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            correct_exe_filter=False,
            print_code_filter=True,
            verbose=verbose,
        )

        # print and correct filters applied
        (
            p_c_percentage,
            p_c_avg_answer_len,
            p_c_norm_avg_answer_len,
            p_c_compressed_avg_len,
            p_c_compressed_norm_avg_len,
            p_c_compressed_seq_avg_len,
            p_c_no_copression_percentage,
            p_c_filtered_df,
        ) = summarize_df_by_filter(
            normalize_df,
            complexity,
            print_code_filter=True,
            correct_exe_filter=True,
            verbose=verbose,
        )

        # Original (no transformation)
        nf_percentage_arr[counter] = nf_percentage
        nf_avg_len_arr[counter] = nf_avg_answer_len
        nf_norm_avg_len_arr[counter] = nf_norm_avg_answer_len
        nf_compr_avg_len_arr[counter] = nf_compressed_avg_len
        nf_compr_norm_avg_len_arr[counter] = nf_compressed_norm_avg_len
        nf_compr_seq_avg_len_arr[counter] = nf_compressed_seq_avg_len
        nf_avg_no_compression_percentage_arr[counter] = nf_no_copression_percentage
        # Print filter
        p_percentage_arr[counter] = p_percentage
        p_avg_len_arr[counter] = p_avg_answer_len
        p_norm_avg_len_arr[counter] = p_norm_avg_answer_len
        p_compr_avg_len_arr[counter] = p_compressed_avg_len
        p_compr_norm_avg_len_arr[counter] = p_compressed_norm_avg_len
        p_compr_seq_avg_len_arr[counter] = p_compressed_seq_avg_len
        p_avg_no_compression_percentage_arr[counter] = p_no_copression_percentage
        # Correct filter
        c_percentage_arr[counter] = c_percentage
        c_avg_len_arr[counter] = c_avg_answer_len
        c_norm_avg_len_arr[counter] = c_norm_avg_answer_len
        c_compr_avg_len_arr[counter] = c_compressed_avg_len
        c_compr_norm_avg_len_arr[counter] = c_compressed_norm_avg_len
        c_compr_seq_avg_len_arr[counter] = c_compressed_seq_avg_len
        c_avg_no_compression_percentage_arr[counter] = c_no_copression_percentage
        # Correct no prints
        cnp_percentage_arr[counter] = cnp_percentage
        cnp_avg_len_arr[counter] = cnp_avg_answer_len
        cnp_norm_avg_len_arr[counter] = cnp_norm_avg_answer_len
        cnp_compr_avg_len_arr[counter] = cnp_compressed_avg_len
        cnp_compr_norm_avg_len_arr[counter] = cnp_compressed_norm_avg_len
        cnp_compr_seq_avg_len_arr[counter] = cnp_compressed_seq_avg_len
        cnp_avg_no_compression_percentage_arr[counter] = cnp_no_copression_percentage
        # incorrect prints
        ip_percentage_arr[counter] = ip_percentage
        ip_avg_len_arr[counter] = ip_avg_answer_len
        ip_norm_avg_len_arr[counter] = ip_norm_avg_answer_len
        ip_compr_avg_len_arr[counter] = ip_compressed_avg_len
        ip_compr_norm_avg_len_arr[counter] = ip_compressed_norm_avg_len
        ip_compr_seq_avg_len_arr[counter] = ip_compressed_seq_avg_len
        ip_avg_no_compression_percentage_arr[counter] = ip_no_copression_percentage
        # Print and Correct filter
        p_c_percentage_arr[counter] = p_c_percentage
        p_c_avg_len_arr[counter] = p_c_avg_answer_len
        p_c_norm_avg_len_arr[counter] = p_c_norm_avg_answer_len
        p_c_compr_avg_len_arr[counter] = p_c_compressed_avg_len
        p_c_compr_norm_avg_len_arr[counter] = p_c_compressed_norm_avg_len
        p_c_compr_seq_avg_len_arr[counter] = p_c_compressed_seq_avg_len
        p_c_avg_no_compression_percentage_arr[counter] = p_c_no_copression_percentage

        complexity_arr.append(complexity)

    output["complexity"] = complexity_arr
    output["language"] = language_col

    # original data (no filters)
    output["percentages"] = nf_percentage_arr
    output["avg_answer_len"] = nf_avg_len_arr
    output["norm_avg_answer_len"] = nf_norm_avg_len_arr
    output["compressed_avg_len"] = nf_compr_avg_len_arr
    output["compressed_norm_avg_len"] = nf_compr_norm_avg_len_arr
    output["compressed_seq_avg_len"] = nf_compr_seq_avg_len_arr
    output["compressed_seq_avg_len"] = nf_compr_seq_avg_len_arr
    output["no_compression_percentage"] = nf_avg_no_compression_percentage_arr
    # only Print filter applied
    output["p_percentages"] = p_percentage_arr
    output["p_avg_answer_len"] = p_avg_len_arr
    output["p_norm_avg_answer_len"] = p_norm_avg_len_arr
    output["p_compressed_avg_len"] = p_compr_avg_len_arr
    output["p_compressed_norm_avg_len"] = p_compr_norm_avg_len_arr
    output["p_compressed_seq_avg_len"] = p_compr_seq_avg_len_arr
    output["p_no_compression_percentage"] = p_avg_no_compression_percentage_arr
    # only Correct executtion filter applied
    output["c_percentages"] = c_percentage_arr
    output["c_avg_answer_len"] = c_avg_len_arr
    output["c_norm_avg_answer_len"] = c_norm_avg_len_arr
    output["c_compressed_avg_len"] = c_compr_avg_len_arr
    output["c_compressed_norm_avg_len"] = c_compr_norm_avg_len_arr
    output["c_compressed_seq_avg_len"] = c_compr_seq_avg_len_arr
    output["c_no_compression_percentage"] = c_avg_no_compression_percentage_arr
    # Correct no prints
    output["cnp_percentages"] = cnp_percentage_arr
    output["cnp_avg_answer_len"] = cnp_avg_len_arr
    output["cnp_norm_avg_answer_len"] = cnp_norm_avg_len_arr
    output["cnp_compressed_avg_len"] = cnp_compr_avg_len_arr
    output["cnp_compressed_norm_avg_len"] = cnp_compr_norm_avg_len_arr
    output["cnp_compressed_seq_avg_len"] = cnp_compr_seq_avg_len_arr
    output["cnp_no_compression_percentage"] = cnp_avg_no_compression_percentage_arr
    # Incorrect prints
    output["ip_percentages"] = ip_percentage_arr
    output["ip_avg_answer_len"] = ip_avg_len_arr
    output["ip_norm_avg_answer_len"] = ip_norm_avg_len_arr
    output["ip_compressed_avg_len"] = ip_compr_avg_len_arr
    output["ip_compressed_norm_avg_len"] = ip_compr_norm_avg_len_arr
    output["ip_compressed_seq_avg_len"] = ip_compr_seq_avg_len_arr
    output["ip_no_compression_percentage"] = ip_avg_no_compression_percentage_arr
    # Print and Correct execution filters applied
    output["p_c_percentages"] = p_c_percentage_arr
    output["p_c_avg_answer_len"] = p_c_avg_len_arr
    output["p_c_norm_avg_answer_len"] = p_c_norm_avg_len_arr
    output["p_c_compressed_avg_len"] = p_c_compr_avg_len_arr
    output["p_c_compressed_norm_avg_len"] = p_c_compr_norm_avg_len_arr
    output["p_c_compressed_seq_avg_len"] = p_c_compr_seq_avg_len_arr
    output["p_c_no_compression_percentage"] = p_c_avg_no_compression_percentage_arr

    language_filtered_df_dic = {
        "nf": nf_filtered_df,
        "p": p_filtered_df,
        "c": c_filtered_df,
        "cnp": cnp_filtered_df,
        "ip": ip_filtered_df,
        "p_c": p_c_filtered_df,
    }

    return output, language_filtered_df_dic


def analyze_all_languages(rough_all_languages_df, verbose=False):
    language_list = list(rough_all_languages_df["language"].unique())
    analysis_df = pd.DataFrame()

    filtered_df_dict = {}

    for l in language_list:
        l_df = copy.deepcopy(
            rough_all_languages_df[rough_all_languages_df["language"] == l]
        )
        language_out, language_filtered_df_dic = create_language_summary_df(
            rough_language_df=l_df, verbose=verbose, save_language_name=l
        )
        # print("***************************************************************")
        # print("LANGUAGE: ", l)
        # print("***************************************************************")
        # print(language_out)
        analysis_df = pd.concat([analysis_df, language_out])
        filtered_df_dict[l] = language_filtered_df_dic

    return analysis_df, filtered_df_dict


def create_plot(df_dict, filter, ax):
    add_title = ""
    # df_dict contains a list of filtered dframes [df, p, c, cnp, ip, pc]
    df = df_dict[filter]
    language = df["language"].unique()
    if len(df) > 0:
        # if no filter, then plot temperature, correct and prints
        if filter == "nf":
            add_title = "Correctboxes and prints in dots"
            # Rename complexity 4 to complexity 3 and remove complexity 3
            df = df[df["complexity"] != 4]
            # correct as boxes
            sns.boxplot(
                x="complexity", y="temperature", hue="correct_execution", data=df, ax=ax
            )
            # prints as dots
            sns.stripplot(
                x="complexity",
                y="temperature",
                hue="print_code",
                data=df,
                jitter=True,
                linewidth=0.5,
                ax=ax,
            )

        ax.set_title("{}. {}".format(language, add_title))
        # Move the legend outside of the plot
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        return None


def plot_temperature(
    all_languages_filtered_df, nf=True, p=False, c=False, cnp=False, ip=False, p_c=False
):
    languages = all_languages_filtered_df.keys()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    plot_index = 0
    for lan in languages:
        lan_filtered_df_dict = all_languages_filtered_df[lan]
        if nf:
            create_plot(lan_filtered_df_dict, "nf", axes[plot_index])
        plot_index += 1
    plt.tight_layout()
    plt.show()


def create_simple_bin_test_pool():
    return [
        "0,0,0,0,0,0,0,0",
        "0,0,0,0,0,0,0,1",
        "0,0,0,0,1,0,0,0",
        "0,0,0,0,0,1,0,0,0",
        "0,0,0,0,0,0,1,1,0",
        "0,0,0,0,0,0,0,1,0",
        "0,0,0,0,0,0,0,1,1",
        "0,0,0,1,1,0,0,0,0",
        "0,0,1,0,0,0,0,0,1",
        "0,1,0,1,0,1,0,1,0",
        "0,0,0,0,1,1,1,0,1",
        "0,0,0,0,0,0,0,1,0,0",
        "0,0,0,0,0,0,0,0,1,0",
        "0,0,0,0,0,0,0,0,1,0",
        "0,0,0,0,0,1,1,0,1,0",
        "0,1,0,1,0,1,0,1,0,1",
        "0,0,1,0,1,0,1,0,1,0",
        "0,1,1,0,1,1,0,1,1,0",
        "0,0,0,0,0,0,0,0,0,1",
        "0,0,1,0,1,0,1,1,0,1",
        "0,1,0,1,0,0,1,0,0,1",
        "0,1,0,1,0,1,1,0,1,0",
        "0,0,0,1,0,1,0,1,0,1,0",
        "0,1,0,1,0,1,0,1,0,1,0",
    ]


def create_ramdom_bin_test_pool():
    return [
        "1,1,0,0,0,1,0,0,0,0,1,0",
        "0,0,0,1,0,1,1,0,1,0,1,1",
        "1,0,0,0,0,1,1,1,1,0,0,1",
        "1,1,1,1,1,1,1,1,0,1,0,0",
        "1,0,1,1,0,1,0,1,0,1,0,0",
        "1,1,1,0,0,1,1,0,1,1,0,1",
        "0,0,1,1,1,0,0,1,0,1,1,0",
        "1,1,1,0,0,0,0,0,0,0,0,0",
        "0,1,1,1,1,0,1,0,1,1,1,1",
        "1,1,0,0,0,1,1,1,0,1,1,1",
        "1,0,0,1,1,0,1,0,0,0,1,0",
        "0,1,1,1,0,1,1,0,1,1,0,1",
        "1,0,1,0,0,1,0,1,0,1,1,1",
        "0,1,1,0,0,1,1,1,1,0,0,1",
        "0,0,1,1,1,1,1,0,1,1,1,0",
        "1,1,1,0,1,1,1,0,0,1,0,0",
        "1,0,1,0,1,0,0,0,0,0,0,0",
        "0,0,1,1,1,1,1,1,0,1,1,1",
        "1,1,0,0,0,1,0,1,0,1,0,1",
        "0,0,0,0,0,1,0,1,1,0,1,0",
        "0,0,1,1,1,1,1,1,0,0,1,1",
        "0,1,0,0,1,0,0,1,0,1,1,1",
        "1,0,0,1,1,0,0,0,0,0,0,1",
        "1,1,0,1,1,1,0,0,1,1,1,0",
        "0,0,0,0,0,0,0,1,0,0,0,0",
        "0,1,0,0,1,1,1,1,1,1,0,1",
        "0,0,1,0,0,0,1,0,0,0,0,0",
        "1,0,0,0,1,1,0,0,1,1,0,0",
        "1,1,0,0,1,0,0,0,0,0,0,0",
        "1,0,1,0,1,0,1,0,0,1,1,1",
        "1,0,1,1,0,1,1,1,1,1,0,0",
        "0,0,0,0,0,0,1,0,0,0,0,0",
        "0,1,1,0,0,0,0,0,0,0,1,1",
        "0,0,1,1,0,1,1,0,1,1,0,1",
        "0,0,0,0,1,1,1,1,1,0,1,1",
        "1,1,1,0,0,0,0,1,1,1,1,1",
        "1,0,1,0,0,1,1,1,1,1,1,1",
        "0,1,0,0,0,0,0,0,1,0,1,0",
        "0,1,0,0,0,1,0,1,1,1,0,0",
        "1,0,1,0,1,1,1,0,1,0,1,0",
        "1,0,0,0,0,1,0,0,1,1,1,1",
        "1,1,0,0,0,0,1,0,0,0,1,1",
        "1,0,1,0,0,0,1,0,1,0,0,0",
        "0,0,1,1,1,1,0,0,1,0,1,0",
        "0,0,0,0,0,0,1,0,0,0,1,0",
        "1,1,1,1,0,0,1,1,1,0,0,1",
        "1,0,1,0,1,1,0,1,0,1,1,0",
        "1,0,0,1,0,1,1,1,1,1,1,0",
        "1,1,1,1,1,0,0,0,1,1,1,0",
        "0,0,0,1,0,1,1,1,0,0,1,1",
        "0,0,1,0,1,0,0,1,0,0,0,0",
        "1,1,1,0,0,1,1,1,0,1,1,1",
        "1,0,0,1,1,1,0,1,1,1,1,1",
        "1,0,0,1,0,0,1,1,0,0,1,1",
        "0,0,0,0,0,0,1,1,0,1,1,1",
        "1,0,0,1,1,0,0,1,1,1,1,0",
        "1,1,1,1,1,0,0,1,0,0,0,1",
        "0,1,1,1,0,1,1,1,1,1,1,1",
        "1,0,1,0,0,1,0,0,1,1,0,1",
        "0,1,0,0,0,1,1,0,0,1,1,1",
        "0,0,0,0,0,1,1,0,1,1,0,1",
        "0,1,0,0,1,0,0,1,0,0,1,0",
        "1,1,1,0,0,1,1,1,1,1,0,1",
        "0,1,0,1,0,1,0,0,1,1,0,1",
        "1,1,1,0,1,0,1,0,0,0,0,1",
        "1,0,0,0,1,0,1,1,1,1,0,1",
        "1,0,1,0,0,0,1,0,1,1,0,1",
        "0,0,1,0,0,1,0,1,1,0,1,1",
        "1,0,0,1,0,1,1,0,0,1,1,0",
        "1,1,0,0,1,1,1,0,0,1,0,1",
        "0,0,1,0,1,1,1,0,0,1,1,1",
        "0,1,1,1,0,1,1,0,1,1,1,0",
        "1,0,0,0,0,1,0,0,1,0,1,0",
        "0,1,1,1,1,0,0,1,0,0,0,0",
        "0,0,0,0,0,1,1,1,0,1,0,0",
        "0,1,0,1,0,1,1,0,0,1,1,0",
        "0,0,0,1,0,0,0,0,1,0,1,0",
        "0,1,1,0,0,0,0,0,1,1,0,1",
        "1,0,0,1,0,1,0,1,1,0,0,0",
        "0,1,0,1,1,1,1,1,0,1,0,1",
        "1,0,0,1,0,0,1,0,1,1,0,0",
        "0,1,0,1,0,1,0,1,1,1,0,1",
        "0,0,1,0,0,1,1,0,1,0,0,1",
        "1,1,0,0,0,0,1,1,1,1,1,0",
        "1,0,1,1,0,0,0,1,0,0,1,0",
        "0,1,1,1,0,0,1,1,0,0,0,1",
        "1,0,1,1,1,0,1,0,1,0,1,0",
        "0,1,0,1,0,0,0,0,0,0,1,0",
        "0,1,0,0,0,1,1,0,1,0,0,0",
        "0,0,1,0,1,0,0,0,1,1,1,1",
        "1,0,1,1,1,0,1,1,1,0,1,1",
        "1,1,1,1,0,1,0,1,0,0,1,0",
        "0,0,1,1,0,0,1,0,1,0,1,0",
        "0,1,1,0,1,0,1,0,1,1,1,0",
        "1,0,1,0,1,0,1,1,0,0,1,0",
        "1,0,1,1,1,1,1,0,1,1,1,0",
        "0,1,0,1,0,0,1,1,0,1,0,1",
        "0,0,0,1,0,1,0,0,1,1,1,1",
        "1,1,1,0,0,0,1,1,1,1,1,1",
        "1,0,0,0,1,1,1,0,1,1,0,1",
    ]


def create_extended_test_pool():
    seq_l1 = [
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40",
        "3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60",
        "4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80",
        "5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100",
        "6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120",
        "7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140",
        "8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160",
        "9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180",
        "10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200",
        "1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39",
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40",
        "11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30",
        "21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40",
        "31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50",
        "41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60",
        "51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70",
        "61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80",
        "71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90",
        "81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100",
        "91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110",
        "101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120",
        "111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130",
        "121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140",
        "131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150",
        "141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160",
        "151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170",
        "161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180",
        "171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190",
        "181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200",
        "191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210",
    ]

    seq_l2 = [
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
        "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288",
        "1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163, 387420489, 1162261467",
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000",
        "1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000",
        "1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210",
        "2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349",
        "0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782, 195025, 470832, 1136689, 2744210, 6625109",
        "1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000, 285311670611, 8916100448256, 302875106592253",  # 11112006825558016, 437893890380859375, 18446744073709551616",  # , 827240261886336764177, 39346408075296537575424, 1978419655660313589123979, 104857600000000000000000000",
        "1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620, 184756, 705432, 2704156, 10400600, 40116600, 155117520, 601080390, 2333606220, 9075135300, 35345263800",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
        "4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57",
        "0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111, 10000, 10001, 10010, 10011",
        "0, 1, 81, 512, 2401, 4913, 5832, 17576, 19683, 234256, 390625, 707281, 1185921, 1382976, 1679616, 4741632, 4826809, 7529536, 11390625, 11529601",
        "1, 2, 145",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89, 104, 120, 137, 155, 174, 194, 215, 237, 260, 284",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110, 111, 118, 119, 121, 128, 129, 130, 131, 138, 139",
        "3, 7, 31, 127, 2047, 8191, 131071, 524287, 8388607, 536870911, 2147483647, 137438953471, 2199023255551, 8796093022207, 140737488355327, 9007199254740991, 576460752303423487, 2305843009213693951",  # 147573952589676412927, 2361183241434822606847",
        "1, 2, 4, 8, 16, 23, 28, 38, 58, 89, 137, 212, 328, 509, 790, 1222, 1891, 2930, 4544, 7048",
        "1, 2, 4, 8, 15, 26, 42, 64, 93, 129, 175, 231, 299, 380, 476, 588, 718, 868, 1040, 1235",
        "1, 5, 12, 22, 35, 51, 70, 92, 117, 145, 176, 210, 247, 287, 330, 376, 425, 477, 532, 590",
        "1, 4, 2, 1, 3, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 0",
        "1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545, 10480142147, 82864869804, 682076806159, 5832742205057, 51724158235372",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
        "1, 11, 21, 1211, 111221, 312211, 13112221, 1113213211, 31131211131221, 13211311123113112211",  # 11131221133112132113212221, 311311222113111231131112132112311321322112111312211312113211, 13211321322113311213211331121113122113121113222112132113213221123113112221131112211312211322211322111312211312111322211322113322113112211331211321113322112111322211322111312211322111312111322211322113322113112211331211131221132211131221121321131211132221121321132132211331121321132211131221131211132211121312211231131122211213211331121321123113213221123113112221131112211312211322211322111312211322111312211322111312111322211322111312211322111322111312211322111312211322111312111322211322111312211322111312211322111312111322211322111312211322111322111312211322111312211322211322111312211322211322113322113112211",
        "5, 7, 11, 23, 47, 59, 83, 107, 167, 179, 227, 263, 347, 359, 383, 467, 479, 503, 563, 587",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288",
        "1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575",
    ]

    seq_l3 = [
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000",
        "2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3",
        "1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89, 105, 122, 140, 159, 179, 200, 222, 245, 269, 294",
        "7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83",
        "1, 11, 111, 1111, 11111, 111111, 1111111, 11111111, 111111111, 1111111111, 11111111111, 111111111111, 1111111111111, 11111111111111, 111111111111111, 1111111111111111, 11111111111111111, 111111111111111111, 1111111111111111111",  # 11111111111111111111",
        "31, 331, 3331, 33331, 333331, 3333331, 33333331, 333333331, 3333333331, 33333333331, 333333333331, 3333333333331, 33333333333331, 333333333333331, 3333333333333331, 33333333333333331, 333333333333333331, 3333333333333333331, 33333333333333333331, 333333333333333333331",
        "1, 3, 7, 13, 21, 31, 43, 57, 73, 91, 111, 133, 157, 183, 211, 241, 273, 307, 343, 381",
        "6, 28, 496, 8128, 130816, 2096128, 33550336, 536870912, 8589869056, 137438691328, 2199023255552, 35184367894528, 562949953421312, 9007199254740992, 144115188075855872, 2305843008139952128",  # 36893488147419103232, 590295810358705651712, 9444732965739290427392, 151115727451828646838272",
        "0, 0, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490",
        "3, 37, 67, 101, 137, 199, 269, 307, 353, 389, 457, 523, 571, 617, 677, 733, 797, 857, 919, 991",
        "1, 2, 145",
        "19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209",
        "2, 3, 7, 43, 1807, 3263443, 10650056950807",  # 113423713055421844361000443, 12864938683278671740537145998360961546653259485195807, 16550664733115248077571638056402000902769537788875589935278839, 274558879151439156701511032526528801",
        "3, 5, 7, 11, 13, 17, 19, 23, 31, 43, 61, 127, 257, 443, 1103, 2203, 4481, 9689, 9941, 11213",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110, 111, 118, 119, 181, 188, 189, 190, 198, 199, 1000",
        "0, 1, 1, 2, 1, 2, 2, 3, 1, 3, 2, 3, 3, 4, 1, 3, 2, 4, 3, 4",
        "0, 1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 12, 6, 18",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
        "1, 4, 2, 1, 3, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 0, 8, 8, 4, 1, 9, 7, 1, 6, 9, 3",
        "1, 7, 3, 2, 0, 5, 0, 7, 9, 8, 1, 4, 0, 9, 7, 8, 2, 7, 7, 0",
        "1, 0, 1, 2, 3, 1, 6, 11, 1, 19, 43, 7, 89, 217, 1, 396, 1003, 143, 1695, 4577",
        "1, 1, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143",
        "3, 5, 11, 17, 31, 41, 59, 67, 83, 109, 127, 157, 179, 191, 211, 241, 277, 283, 353, 367",
        "1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28, 36, 38, 47, 48, 53, 57, 62, 69",
        "1, 0, 1, 1, 2, 3, 6, 11, 23, 46, 98, 207, 451, 983, 2179, 4850, 10905, 24631, 55931, 127335",
        "1093, 3511, 10501, 10961, 13759, 20771, 22777, 27541, 29341, 30269, 33233, 42641, 52639, 55271, 65617, 69109, 81041, 82507, 85831, 100517",
        "5, 13, 563",
    ]
    seq_l4 = [
        "29, 57, 68, 120, 134, 140, 173, 197, 283, 313, 331, 342, 360, 385, 397, 404, 432, 465, 487, 498",
        "24, 26, 36, 40, 184, 226, 244, 384, 391, 423, 439, 446, 462, 466, 470, 473, 476, 480, 487, 496",
        "90, 203, 212, 235, 270, 324, 342, 352, 371, 417, 420, 422, 431, 434, 439, 446, 461, 473, 495, 497",
        "20, 48, 95, 234, 282, 296, 352, 402, 428, 481, 493, 497, 501, 507, 509, 512, 530, 544, 556, 590",
        "62, 98, 130, 154, 290, 315, 324, 385, 408, 447, 449, 451, 455, 462, 478, 479, 489, 491, 498, 499",
        "2, 42, 66, 102, 153, 195, 201, 252, 306, 396, 402, 414, 420, 427, 429, 438, 442, 451, 472, 486",
        "128, 151, 153, 217, 224, 332, 382, 400, 450, 478, 488, 490, 494, 496, 500, 503, 505, 509, 512, 520",
        "26, 50, 114, 148, 160, 170, 274, 347, 432, 497, 502, 511, 519, 524, 535, 543, 556, 572, 580, 593",
        "48, 94, 176, 177, 219, 276, 282, 283, 459, 488, 489, 492, 494, 499, 500, 504, 507, 512, 517, 519",
        "139, 252, 272, 281, 304, 361, 370, 415, 438, 500, 503, 505, 511, 513, 515, 517, 522, 526, 529, 536",
        "15, 95, 115, 195, 240, 318, 326, 350, 432, 450, 453, 455, 466, 469, 472, 480, 485, 489, 493, 498",
        "134, 224, 293, 378, 379, 395, 434, 451, 482, 496, 499, 502, 504, 509, 515, 519, 522, 526, 529, 535",
        "23, 93, 142, 145, 245, 266, 296, 317, 428, 495, 496, 502, 507, 512, 516, 521, 526, 531, 534, 540",
        "18, 39, 71, 194, 197, 219, 263, 270, 416, 473, 478, 483, 486, 489, 492, 497, 502, 505, 507, 511",
        "9, 84, 144, 170, 325, 393, 401, 405, 435, 497, 498, 499, 503, 506, 511, 513, 517, 519, 522, 525",
        "26, 40, 202, 267, 282, 340, 359, 408, 410, 495, 497, 502, 505, 507, 509, 511, 515, 518, 520, 525",
        "34, 92, 164, 165, 209, 296, 414, 456, 467, 494, 499, 503, 505, 507, 510, 514, 517, 521, 523, 529",
        "16, 119, 121, 123, 135, 139, 285, 311, 409, 412, 413, 416, 420, 424, 429, 434, 439, 443, 447, 450",
        "8, 11, 12, 103, 116, 196, 247, 254, 389, 427, 433, 439, 444, 446, 448, 450, 453, 457, 460, 464",
        "12, 36, 96, 119, 171, 213, 221, 232, 363, 451, 457, 463, 466, 471, 475, 480, 483, 487, 490, 493",
        "38, 91, 142, 197, 215, 313, 316, 319, 423, 466, 468, 473, 478, 481, 485, 489, 491, 495, 498, 500",
        "7, 42, 147, 201, 213, 248, 310, 332, 436, 479, 483, 487, 490, 493, 495, 497, 501, 505, 508, 511",
        "27, 101, 105, 164, 245, 290, 304, 441, 449, 490, 492, 496, 501, 503, 505, 507, 511, 514, 516, 520",
        "4, 11, 29, 106, 214, 283, 296, 298, 360, 497, 498, 501, 503, 506, 509, 511, 513, 515, 517, 520",
        "72, 106, 139, 165, 171, 192, 199, 429, 453, 477, 479, 482, 484, 487, 491, 493, 495, 497, 499, 502",
        "187, 218, 260, 295, 301, 314, 379, 410, 452, 469, 472, 474, 476, 478, 481, 483, 485, 487, 489, 491",
        "29, 63, 95, 140, 150, 190, 221, 437, 482, 491, 497, 499, 501, 503, 505, 507, 509, 511, 513, 515",
        "3, 11, 84, 144, 156, 177, 188, 199, 229, 284, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335",
        "26, 94, 98, 137, 176, 301, 323, 330, 372, 444, 448, 453, 456, 459, 462, 465, 468, 471, 474, 477",
        "39, 81, 88, 210, 215, 378, 416, 430, 439, 490, 493, 495, 498, 501, 503, 505, 507, 509, 511, 513",
    ]

    return [
        seq_l1,
        seq_l2,
        seq_l3,
        seq_l4,
    ]


def create_original_length_test_pool():
    original_l1 = [
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20",
        "3, 6, 9, 12, 15, 18, 21, 24, 27, 30",
        "4, 8, 12, 16, 20, 24, 28, 32, 36, 40",
        "5, 10, 15, 20, 25, 30, 35, 40, 45, 50",
        "6, 12, 18, 24, 30, 36, 42, 48, 54, 60",
        "7, 14, 21, 28, 35, 42, 49, 56, 63, 70",
        "8, 16, 24, 32, 40, 48, 56, 64, 72, 80",
        "9, 18, 27, 36, 45, 54, 63, 72, 81, 90",
        "10, 20, 30, 40, 50, 60, 70, 80, 90, 100",
        "1, 3, 5, 7, 9, 11, 13, 15, 17, 19",
        "2, 4, 6, 8, 10, 12, 14, 16, 18, 20",
        "11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
        "21, 22, 23, 24, 25, 26, 27, 28, 29, 30",
        "31, 32, 33, 34, 35, 36, 37, 38, 39, 40",
        "41, 42, 43, 44, 45, 46, 47, 48, 49, 50",
        "51, 52, 53, 54, 55, 56, 57, 58, 59, 60",
        "61, 62, 63, 64, 65, 66, 67, 68, 69, 70",
        "71, 72, 73, 74, 75, 76, 77, 78, 79, 80",
        "81, 82, 83, 84, 85, 86, 87, 88, 89, 90",
        "91, 92, 93, 94, 95, 96, 97, 98, 99, 100",
        "101, 102, 103, 104, 105, 106, 107, 108, 109, 110",
        "111, 112, 113, 114, 115, 116, 117, 118, 119, 120",
        "121, 122, 123, 124, 125, 126, 127, 128, 129, 130",
        "131, 132, 133, 134, 135, 136, 137, 138, 139, 140",
        "141, 142, 143, 144, 145, 146, 147, 148, 149, 150",
        "151, 152, 153, 154, 155, 156, 157, 158, 159, 160",
        "161, 162, 163, 164, 165, 166, 167, 168, 169, 170",
        "171, 172, 173, 174, 175, 176, 177, 178, 179, 180",
        "181, 182, 183, 184, 185, 186, 187, 188, 189, 190",
        "191, 192, 193, 194, 195, 196, 197, 198, 199, 200",
    ]

    original_l2 = [
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 1, 2, 3, 5, 8, 13, 21, 34, 55",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512",
        "1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683",
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000",
        "1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880",
        "1, 3, 6, 10, 15, 21, 28, 36, 45, 55",
        "2, 1, 3, 4, 7, 11, 18, 29, 47, 76",
        "0, 1, 2, 5, 12, 29, 70, 169, 408, 985",
        "1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 10000000000",
        "1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "4, 6, 9, 10, 14, 15, 21, 22, 25, 26",
        "1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010",
        "0, 1, 81, 512, 2401, 4913, 5832, 17576, 19683, 234256",
        "1, 2, 145, 40585",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110",
        "3, 7, 31, 127, 2047, 8191, 131071, 524287, 8388607, 536870911",
        "1, 2, 4, 8, 16, 23, 28, 38, 58, 89",
        "1, 2, 4, 8, 15, 26, 42, 64, 93, 129",
        "1, 5, 12, 22, 35, 51, 70, 92, 117, 145",
        "0, 1, 1, 2, 1, 2, 2, 3, 1, 3",
        "1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 11, 21, 1211, 111221",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 2, 4, 8, 16, 32, 64, 128, 256, 512",
        "1, 3, 7, 15, 31, 63, 127, 255, 511, 1023",
    ]

    original_l3 = [
        "1, 4, 9, 16, 25, 36, 49, 64, 81, 100",
        "1, 8, 27, 64, 125, 216, 343, 512, 729, 1000",
        "2, 1, 2, 1, 2, 2, 1, 2, 2, 2",
        "1, 2, 2, 3, 3, 4, 4, 4, 5, 5",
        "2, 4, 8, 16, 32, 64, 128, 256, 512, 1024",
        "2, 5, 12, 20, 29, 39, 50, 62, 75, 89",
        "7, 11, 13, 17, 19, 23, 29, 31, 37, 41",
        "1, 11, 111, 1111, 11111, 111111, 1111111, 11111111, 111111111, 1111111111",
        "31, 331, 3331, 33331, 333331, 3333331, 33333331, 333333331",
        "1, 3, 7, 13, 21, 31, 43, 57, 73, 91",
        "6, 28, 496, 8128, 130816, 2096128",
        "0, 0, 2, 3, 5, 7, 11, 15, 22, 30",
        "3, 37, 67, 101, 137, 199, 269, 307, 353, 389",
        "1, 2, 145, 40585",
        "19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203",
        "2, 3, 7, 42, 1806, 3263442",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 8, 10, 18, 19, 100, 101, 108, 109, 110",
        "0, 1, 1, 2, 1, 2, 2, 3, 1, 3",
        "0, 1, 1, 2, 2, 4, 2, 6, 4, 6",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "1, 4, 2, 1, 3, 5, 9, 2, 6, 5",
        "1, 7, 3, 2, 0, 5, 0, 7, 9, 8",
        "1, 0, 1, 2, 3, 1, 6, 11",
        "1, 1, 1, 3, 7, 15, 31, 63",
        "3, 5, 11, 17, 31, 41, 59, 67",
        "1, 2, 3, 4, 6, 8, 11, 13",
        "1, 0, 1, 1, 2, 3, 6, 11",
        "1093, 3511",
        "5, 13, 563",
    ]
    original_l4 = [
        "29, 57, 68, 120, 134, 140, 173, 197, 283, 313",
        "24, 26, 36, 40, 184, 226, 244, 384, 391, 423",
        "90, 203, 212, 235, 270, 324, 342, 352, 371, 417",
        "20, 48, 95, 234, 282, 296, 352, 402, 428, 481",
        "62, 98, 130, 154, 290, 315, 324, 385, 408, 447",
        "2, 42, 66, 102, 153, 195, 201, 252, 306, 396",
        "128, 151, 153, 217, 224, 332, 382, 400, 450, 478",
        "26, 50, 114, 148, 160, 170, 274, 347, 432, 497",
        "48, 94, 176, 177, 219, 276, 282, 283, 459, 488",
        "139, 252, 272, 281, 304, 361, 370, 415, 438, 500",
        "15, 95, 115, 195, 240, 318, 326, 350, 432, 450",
        "134, 224, 293, 378, 379, 395, 434, 451, 482, 496",
        "23, 93, 142, 145, 245, 266, 296, 317, 428, 495",
        "18, 39, 71, 194, 197, 219, 263, 270, 416, 473",
        "9, 84, 144, 170, 325, 393, 401, 405, 435, 497",
        "26, 40, 202, 267, 282, 340, 359, 408, 410, 495",
        "34, 92, 164, 165, 209, 296, 414, 456, 467, 494",
        "16, 119, 121, 123, 135, 139, 285, 311, 409, 412",
        "8, 11, 12, 103, 116, 196, 247, 254, 389, 427",
        "12, 36, 96, 119, 171, 213, 221, 232, 363, 451",
        "38, 91, 142, 197, 215, 313, 316, 319, 423, 466",
        "7, 42, 147, 201, 213, 248, 310, 332, 436, 479",
        "27, 101, 105, 164, 245, 290, 304, 441, 449, 490",
        "4, 11, 29, 106, 214, 283, 296, 298, 360, 497",
        "72, 106, 139, 165, 171, 192, 199, 429, 453, 477",
        "187, 218, 260, 295, 301, 314, 379, 410, 452, 469",
        "29, 63, 95, 140, 150, 190, 221, 437, 482, 491",
        "3, 11, 84, 144, 156, 177, 188, 199, 229, 284",
        "26, 94, 98, 137, 176, 301, 323, 330, 372, 444",
        "39, 81, 88, 210, 215, 378, 416, 430, 439, 490",
    ]

    return [original_l1, original_l2, original_l3, original_l4]
