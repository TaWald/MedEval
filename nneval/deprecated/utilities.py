import csv
import os
import json

def save_list_dict_to_csv(listdict: list[dict], output_path: str, filename: str):
    """Write the listdict to os.path.join(output_path, filename)

    :param listdict:
    :param output_path:
    :param filename:
    :return:
    """
    keys = list(listdict[0].keys())

    if filename.endswith(".csv"):
        pass
    else:
        filename = filename + ".csv"

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, filename), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(listdict)
    return
