import time
from typing import List

import torch
from ignite.engine import Events
from prettytable import PrettyTable

from ..module import Module


class ScreenLogger(Module):
    def __init__(self, classes: List[str], eval_names: List[str] = None):
        super(ScreenLogger, self).__init__()
        self.classes = classes
        self.eval_names = eval_names if eval_names else []

    def init(self):
        assert 'logger' in self.frame, 'The frame does not have logger.'
        self.model = self.frame['model']
        self.writer = self.frame.get('writer', None)
        self.logger = self.frame['logger']

        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.STARTED, self._started)
        self.frame['engine'].engine.add_event_handler(Events.COMPLETED, self._completed)

        if len(self.eval_names):
            assert 'metrics' in self.frame, 'The frame does not have metrics.'
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_screen)

    def _started(self, engine):
        self.logger.info(
            f'Model Params: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)} params.'
        )
        self.logger.info(f'{time.asctime()} - STARTED')

    def _completed(self, engine):
        self.logger.info(f'{time.asctime()} - COMPLETED')

    def _log_screen(self, engine):
        self.logger.info('')
        self.logger.info(f'Epoch #{engine.state.epoch} - {time.asctime()}')

        for eval_name in self.eval_names:
            self.logger.info(f'<{eval_name}>')
            for metric_name, metric_value in self.frame['metrics'].metric_values[eval_name].items():
                if isinstance(metric_value, float):
                    self.logger.info(f'\t*{metric_name}: {metric_value:.5f}')
                    if self.writer is not None:
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict={eval_name: metric_value},
                            global_step=engine.state.epoch
                        )
                elif isinstance(metric_value, torch.Tensor):
                    metric_values = metric_value.detach().cpu().numpy().tolist()
                    if len(self.classes) == len(metric_values) - 1:
                        metric_values = metric_values[1:]

                    # metric = PrettyTable(['Field Name', metric_name.upper()])  # heading of table
                    # for class_name, metric_value in zip(self.classes, metric_values):
                    #     metric.add_row([class_name, metric_value])

                    #     if self.writer is not None:
                    #         self.writer.add_scalars(
                    #             main_tag=eval_name,
                    #             tag_scalar_dict={f'{metric_name}_{class_name}': metric_value},
                    #             global_step=engine.state.epoch
                    #         )

                    # self.logger.info(f'\t*{metric_name}')
                    # self.logger.info(metric)
