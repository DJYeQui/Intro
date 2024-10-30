import csv


def csv_file_reader(path):
    with open(path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            print(row)

csv_file_reader("D:\GitDeskopProject\RAGIntro\job_descriptions.csv")