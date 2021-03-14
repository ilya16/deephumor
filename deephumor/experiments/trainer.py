import os
import sys
import time
from datetime import datetime

import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import perplexity
from deephumor.models.utils import get_mask_from_lengths


class Trainer:
    """An ultimate class for training the models."""
    def __init__(self, experiment_title, log_dir='./logs', text_labels=False,
                 phases=('train', 'val'), grad_clip_norm=1., fp16_run=True, device='cuda'):
        self.experiment_data = self._setup_experiment(experiment_title, log_dir)

        self.text_labels = text_labels
        self.phases = phases
        self.grad_clip_norm = grad_clip_norm
        self.fp16_run = fp16_run
        self.device = device

        self.writers = self._setup_writers()

    @staticmethod
    def _setup_experiment(title, log_dir='./logs'):
        experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
        experiment_dir = os.path.join(log_dir, experiment_name)
        best_model_path = os.path.join(experiment_dir, f"{title}.best.pth")

        experiment_data = {
            'model_name': title,
            'name': experiment_name,
            'dir': experiment_dir,
            'best_model_path': best_model_path,
            'epochs': 0,
            'iterations': 0,
        }

        return experiment_data

    def _setup_writers(self):
        return {
            phase: SummaryWriter(log_dir=os.path.join(self.experiment_data['dir'], phase))
            for phase in self.phases
        }

    def _to_device(self, tensor):
        return tensor.to(self.device, non_blocking=True)

    def run_epoch(self, model, dataloader, optimizer, criterion, scaler=None, phase='train'):
        is_train = (phase == 'train')
        model.train() if is_train else model.eval()

        epoch = self.experiment_data['epochs']
        iterations = self.experiment_data['iterations']
        num_samples, epoch_loss, epoch_pp = 0, 0., 0.

        with torch.set_grad_enabled(is_train):
            epoch_pbar = tqdm(dataloader, position=0, leave=False, file=sys.stdout)
            for batch in epoch_pbar:
                # unpack batch
                labels, captions, images, lengths = batch
                batch_size = captions.size(0)

                captions, images, lengths = map(self._to_device, (captions, images, lengths))

                with amp.autocast(enabled=self.fp16_run):
                    if self.text_labels:
                        labels = self._to_device(labels)
                        pred = model(images, captions[:, :-1], lengths, labels)
                    else:
                        pred = model(images, captions[:, :-1], lengths)

                    mask = get_mask_from_lengths(lengths)
                    loss = criterion(pred[mask], captions[mask])

                with torch.no_grad():
                    pp = perplexity(pred, captions, lengths)

                if is_train:
                    iterations += 1

                if self.writers is not None and phase in self.writers and is_train:
                    # make optimization step
                    optimizer.zero_grad()

                    if scaler is None:
                        loss.backward()
                    else:
                        scaler.scale(loss).backward()

                    if self.grad_clip_norm > 0.:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                        self.writers[phase].add_scalar(f"train/grad_norm", grad_norm, iterations)

                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()

                loss, pp = loss.item(), pp.item()
                epoch_loss += loss * batch_size
                epoch_pp += pp * batch_size
                num_samples += batch_size

                # dump batch metrics to tensorboard
                if self.writers is not None and phase in self.writers and is_train:
                    self.writers[phase].add_scalar(f"train/batch_loss", loss, iterations)
                    self.writers[phase].add_scalar(f"train/batch_perplexity", pp, iterations)

                epoch_pbar.set_description(f"loss: {epoch_loss / num_samples:.5f}, "
                                           f"pp: {epoch_pp / num_samples:.5f}")

            epoch_pbar.close()
            epoch_loss = epoch_loss / len(dataloader.dataset)
            epoch_pp = epoch_pp / len(dataloader.dataset)

            # dump epoch metrics to tensorboard
            if self.writers is not None and phase in self.writers:
                self.writers[phase].add_scalar(f"eval/loss", epoch_loss, epoch)
                self.writers[phase].add_scalar(f"eval/perplexity", epoch_pp, epoch)

        if is_train:
            self.experiment_data['iterations'] = iterations

        return epoch_loss, epoch_pp

    def train_model(self, model, dataloaders, optimizer, criterion, scheduler=None, n_epochs=50):

        best_epoch, best_val_loss = 0, float('+inf')
        past_epochs = self.experiment_data['epochs']

        scaler = amp.GradScaler(enabled=self.fp16_run)

        if self.writers is None:
            self._setup_writers()

        for epoch in range(past_epochs + 1, past_epochs + n_epochs + 1):
            self.experiment_data['epochs'] = epoch
            print(f'Epoch {epoch:02d}/{past_epochs + n_epochs:02d}')

            st = time.perf_counter()
            for phase in self.phases:
                epoch_loss, epoch_pp = self.run_epoch(
                    model, dataloaders[phase], optimizer, criterion, scaler=scaler, phase=phase
                )
                print(f'  {phase:5s} loss: {epoch_loss:.5f}, perplexity: {epoch_pp:.3f}')

                if phase == 'val' and epoch_loss < best_val_loss:
                    best_epoch, best_val_loss = epoch, epoch_loss
                    model.save(self.experiment_data['best_model_path'])

                model.save(f"{self.experiment_data['model_name']}.e{epoch}.pth")

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

            et = time.perf_counter() - st
            print(f'  epoch time: {et:.2f}s')

        print(f'Best val_loss: {best_val_loss} (epoch: {best_epoch})')

        return self.experiment_data

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers = None
