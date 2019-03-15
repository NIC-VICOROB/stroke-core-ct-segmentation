# Time of logging
import csv


class TrainingLogger:
    def __init__(self):
        self.epoch_params = []

    def add_epoch_params(self, epoch_params):
        assert isinstance(epoch_params, dict)
        self.epoch_params.append(epoch_params)
        if len(epoch_params) > 1:
            # TODO check that all dictionaries have same keys
            pass

    def write_to_csv(self, filepath):
        with open(filepath, 'w') as csvfile:
            fieldnames = [param_name for param_name in self.epoch_params[0].keys()]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for epoch_params in self.epoch_params:
                writer.writerow(epoch_params)
            del writer