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
        fileExists = False
        try:
            with open(self.filePath, "r") as file:
                fileExists = True
        except FileNotFoundError:
            pass

        with open(self.filePath, "a", newline="") as file:
            fieldnames = list(metricsDict.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not fileExists:
                writer.writeheader()

            writer.writerow(metricsDict)
