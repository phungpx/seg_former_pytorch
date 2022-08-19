from ..module import Module
from ignite.engine import Events
from torch.optim import lr_scheduler


class LRScheduler(Module):
    def __init__(self, score_name, evaluator_name, scheduler):
        super(LRScheduler, self).__init__()
        self.score_name = score_name
        self.scheduler = scheduler
        self.evaluator_name = evaluator_name

    def init(self):
        # assert 'optim' in self.frame, 'The frame does not have optim.'
        # self.scheduler = self.lr_scheduler(self.frame['optim'], **self.kwargs)
        self.frame["engine"].engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._step
        )

    def _step(self, engine):
        self.scheduler.step(
            self.frame[self.evaluator_name].engine.state.metrics[self.score_name]
        )

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class ReduceLROnPlateau(Module):
    def __init__(self, score_name, evaluator_name, **kwargs):
        super(ReduceLROnPlateau, self).__init__()
        self.score_name = score_name
        self.evaluator_name = evaluator_name
        self.kwargs = kwargs

    def init(self):
        assert "optim" in self.frame, "The frame does not have optim."
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.frame["optim"], **self.kwargs
        )
        self.frame["engine"].engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._step
        )

    def _step(self, engine):
        self.scheduler.step(
            self.frame[self.evaluator_name].engine.state.metrics[self.score_name]
        )

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class MultiStepLR(Module):
    def __init__(self, score_name, evaluator_name, **kwargs):
        super(ReduceLROnPlateau, self).__init__()
        self.score_name = score_name
        self.evaluator_name = evaluator_name
        self.kwargs = kwargs

    def init(self):
        assert "optim" in self.frame, "The frame does not have optim."
        self.scheduler = lr_scheduler.MultiStepLR(self.frame["optim"], **self.kwargs)
        self.frame["engine"].engine.add_event_handler(
            Events.EPOCH_COMPLETED, self._step
        )

    def _step(self, engine):
        self.scheduler.step(
            self.frame[self.evaluator_name].engine.state.metrics[self.score_name]
        )

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
