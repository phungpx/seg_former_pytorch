from abc import abstractmethod

import torch
from ignite import engine as e

from ...module import Module

__all__ = ['Engine', 'Trainer', 'Evaluator']


class Engine(Module):
    '''
        Base class for all engines. Your engine should subclass this class.
        Class Engine contains an Ignite Engine that controls running process over a dataset.
        Method _update is a function receiving the running Ignite Engine
        and the current batch in each iteration and returns data to be stored in the Ignite Engine's state.
        Parameters:
            dataset_name (str): dataset which engine run over.
            device (str): device on which model and tensor is allocated.
            max_epochs (int): number of epochs training process runs.
    '''

    def __init__(self, dataset, device, max_epochs=1):
        super(Engine, self).__init__()
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.engine = e.DeterministicEngine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    @abstractmethod
    def _update(self, engine, batch):
        pass


class Trainer(Engine):
    '''
        Engine controls training process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'loss' in self.frame, 'The frame does not have loss.'
        assert 'logger' in self.frame, 'The frame does not have logger.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']
        self.logger = self.frame['logger']
        self.scaler = self.frame.get('scaler', None)  # https://pytorch.org/docs/stable/amp.html
        self.writer = self.frame.get('writer', None)
        if self.scaler is not None:
            self.logger.info('Applied FP16 mode.')

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]

        # casts operations to mixed precision
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            params[0] = self.model(params[0])
            loss = self.loss(*params)

        if self.scaler is not None:
            # scales the loss, and calls backward() to create scaled gradients
            self.scaler.scale(loss).backward()
            # unscale gradients and calls or skips optimizer.step()
            self.scaler.step(self.optimizer)
            # update the scale for next iteration
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.writer is not None:
            step = engine.state.iteration
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(tag='learning_rate', scalar_value=lr, global_step=step)
            self.writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)

        return loss.item()


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            params[0] = self.model(params[0])
            return params
