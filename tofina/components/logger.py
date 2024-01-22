import csv
from typing import Callable, List, Mapping
import torch


class Logger:
    def __init__(self) -> None:
        pass

    def processRecord(
        self, metricsDict: Mapping[str, Callable[[List[torch.Tensor]], float]], t: int
    ) -> None:
        pass


class TensorBoardLogger(Logger):
    def __init__(self) -> None:
        pass


def fileExists(filePath: str) -> bool:
    try:
        with open(filePath, "r") as file:
            return True
    except FileNotFoundError:
        return False


class CsvLogger(Logger):
    def __init__(self, filePath: str) -> None:
        if fileExists(filePath):
            raise FileExistsError("File already exists")
        self.filePath = filePath

    def processRecord(
        self, metricsDict: Mapping[str, Callable[[List[torch.Tensor]], float]], t: int
    ) -> None:
        # Check if the file exists
        fileExists = False
        try:
            with open(self.filePath, "r") as file:
                fileExists = True
        except FileNotFoundError:
            pass

        # Open the CSV file in append mode and write the header if the file is being created
        with open(self.filePath, "a", newline="") as file:
            fieldnames = list(metricsDict.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if the file is being created
            if not fileExists:
                writer.writeheader()

            # Write the data
            writer.writerow(metricsDict)
