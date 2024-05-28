import time
import torch

class LayerLogger:
    """A logger for layer-specific logging in csv format.
    Args:
        output_file: The file to log to.
        headers: List of headers for the csv.
    """

    def __init__(
        self,
        output_file,
        headers = None
    ) -> None:
        self.folder = '/home/theoag/cse552/mixtral_parallel/batch_size_1'
        self.output_file = f'{self.folder}/{output_file}'
        self.start_time = 0
        self.headers = headers
        if headers:
            self.write_headers()

    def write_headers(self) -> None:
        file = open(self.output_file, 'a')
        log_line = ','.join(self.headers)
        file.write(log_line + '\n')
        file.close()

    def write(
        self,
        *args
    ) -> None:
        file = open(self.output_file, 'a')
        log_line = ','.join([str(arg) for arg in args])
        file.write(log_line + '\n')
        file.close()

    def start_timer(self) -> None:
        torch.cuda.synchronize()
        self.start_time = time.perf_counter_ns()

    def get_timer_value(
        self,
        units = 'ns'
    ) -> int:
        torch.cuda.synchronize()
        diff_ns = time.perf_counter_ns() - self.start_time

        if units == 'ns':
            return diff_ns
        elif units == 'us':
            return diff_ns / 1000.0
        elif units == 'ms':
            return diff_ns / 1000000.0
        elif units == 's':
            return diff_ns / 1000000000.0
        else:
            raise Exception('Unit not recognized')