import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import torchaudio
import numpy as np
import recognizer.tools as tools
import recognizer.hmm as HMM
import edit_distance
from typing import Optional, Callable

import multiprocessing


# dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
#                                                     window_fn=window_fn, pad_mode=pad_mode, onesided=onesided)

def parallel_decoding(data):
    posteriors, true_length, text, hmm = data
    posteriors = posteriors[:true_length]

    best_path, pstar = hmm.viterbi_decode(posteriors)
    word_seq = hmm.getTranscription(best_path)
    ref_seq = text.split(' ')

    if word_seq == -1:
        # This means our HMM could not decode!
        word_seq = ['']
    # edit distance
    res = edit_distance.SequenceMatcher(a=ref_seq, b=word_seq)

    return word_seq, best_path, pstar, res.distance()


def complex_to_mag(complex_spectrum_in_real):
    if len(complex_spectrum_in_real.shape) == 4:
        complex_spectrum = complex_spectrum_in_real[:, :, :, 0] + complex_spectrum_in_real[:, :, :, 1] * 1j
    elif len(complex_spectrum_in_real.shape) == 3:
        complex_spectrum = complex_spectrum_in_real[:, :, 0] + complex_spectrum_in_real[:, :, 1] * 1j
    return complex_spectrum.abs()


class MySpectrogram(torch.nn.Module):

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(MySpectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        # self.pad = pad
        # self.power = power
        self.normalized = normalized
        self.pad_mode = 'constant'

    def forward(self, waveform):
        r"""
               Args:
                   waveform (Tensor): Tensor of audio of dimension (..., time).

               Returns:
                   Tensor: Dimension (..., freq, time), where freq is
                   ``n_fft // 2 + 1`` where ``n_fft`` is the number of
                   Fourier bins, and time is the number of window hops (n_frame).
        """
        # pack batch
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        # default values are consistent with librosa.core.spectrum._spectrogram
        spec_f = torch.stft(
            waveform, self.n_fft, self.hop_length, self.win_length, self.window, center=True, pad_mode="constant",
            normalized=False, onesided=True, return_complex=True
        )

        # unpack batch
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

        spec_f = spec_f.abs()

        return spec_f


class MyMFCC(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_mels: bool = False,
                 speckwargs: Optional[dict] = None,
                 melscalekwargs: Optional[dict] = None) -> None:
        super(MyMFCC, self).__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.log_mels = log_mels

        self.spectrogram = MySpectrogram(**speckwargs)
        self.mel_scale = torchaudio.transforms.MelScale(**melscalekwargs)

        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB('power', self.top_db)

        dct_mat = torchaudio.functional.create_dct(self.n_mfcc, self.mel_scale.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, waveform: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        specgram = self.spectrogram(waveform)
        specgram_pow_2 = torch.pow(specgram, 2.0)
        mel_specgram = self.mel_scale(specgram_pow_2)

        if self.log_mels:
            assert False, "We set this parameter to False"
            # log_offset = 1e-6
            # mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram_db = self.amplitude_to_DB(mel_specgram)

        # (..., channel, n_mels, time).transpose(...) dot (n_mels, n_mfcc)
        # -> (..., channel, time, n_mfcc).transpose(...)
        mfcc = torch.matmul(mel_specgram_db.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
        return specgram, mfcc


class BaseModel(nn.Module):

    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(BaseModel, self).__init__()

        self.feature_parameters = feature_parameters
        self.hop_size_samples = self.feature_parameters['hop_size_samples']
        self.window_size_samples = self.feature_parameters['window_size_samples']
        self.left_context = feature_parameters['left_context']
        self.right_context = feature_parameters['right_context']
        self.n_mfcc = feature_parameters['num_ceps']
        self.dropout = dropout
        self.hmm = hmm
        # self.spectrogram_pad_mode = 'constant'

        self.test_dropout_enabled = test_dropout_enabled

        self.mfcc = MyMFCC(n_mfcc=self.n_mfcc,
                           speckwargs=dict(n_fft=self.window_size_samples, hop_length=self.hop_size_samples,
                                           win_length=self.window_size_samples, window_fn=torch.hamming_window,
                                           # pad_mode=self.spectrogram_pad_mode, onesided=True
                                           # pad=0, power=None,
                                           ),
                           melscalekwargs=dict(n_stft=self.window_size_samples // 2 + 1))

        # delta and deltadeltas
        self.deltas = torchaudio.transforms.ComputeDeltas()

    def save(self, model_path):
        torch.save(self, model_path)

    @staticmethod
    def load(model_path):
        return torch.load(model_path).cuda()

    def forward(self, x, penu=False):
        pass

    def train_model(self, dataset, lr=0.0001, epochs=20, batch_size=64, viterbi_training=False, update_y_label=False,
                    rand_obj=None, print_progress=True):

        lam = 1 / 1505426  # weight decay

        opt = optim.Adam(self.parameters(), lr, weight_decay=lam)
        CrEnt = nn.CrossEntropyLoss()

        losses = []
        # progress bar
        with tqdm(total=len(dataset.X), bar_format='    {l_bar}{bar:30}{r_bar}', disable=not print_progress) as pbar:
            for n_iter in range(epochs):
                ac_loss = 0
                num_frames = 0

                if viterbi_training:
                    # init A for viterbi training
                    self.hmm.A_count = np.ceil(self.hmm.A)

                for local_x, local_y, text, y_true_length, filenames in dataset.generator(batch_size=batch_size,
                                                                                          return_filename=True,
                                                                                          shuffle=True,
                                                                                          rand_obj=rand_obj):

                    if viterbi_training:
                        posteriors = self.features_to_posteriors(local_x, is_training=True)
                        local_y = self.hmm.viterbi_train(posteriors, y_true_length, local_y, text)
                        if update_y_label:
                            dataset.update_y_label([y[:t] for y, t in zip(local_y, y_true_length)], filenames)

                    if batch_size == 1:
                        loss = self.compute_loss_single(local_x, local_y, is_training=True)
                    else:
                        loss = self.compute_loss_batch(local_x, local_y, y_true_length, is_training=True)

                    ac_loss += loss.item() * local_y.shape[0]
                    num_frames += local_y.shape[0]

                    loss.backward()
                    # nn.utils.clip_grad_value_(self.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    torch.cuda.empty_cache()
                
                    pbar.set_description(f'Epoch {n_iter + 1}/{epochs}: loss {ac_loss/num_frames:.6f})')
                    pbar.update(local_y.shape[0])

                # normalize to loss per frame
                av_loss = ac_loss / num_frames

                # update progress bar
                # pbar.set_description(f'Epochs {n_iter + 1}/{epochs} (loss {av_loss:.6f})')
                # pbar.update(1)

                losses.append(av_loss)

                if print_progress:
                    pbar.reset()

        return losses

    def test(self, dataset):
        # init stats
        E, N = 0, 0
        with tqdm(total=len(dataset), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            for x, y, text, y_true_length in dataset.generator():
                posteriors = self.features_to_posteriors(x)
                # run viterbi to get recognized words
                best_path, pstar = self.hmm.viterbi_decode(posteriors)
                word_seq = self.hmm.getTranscription(best_path)
                # get original text
                ref_seq = text.split(' ')

                if word_seq == -1:
                    word_seq = ['']
                # edit distance
                res = edit_distance.SequenceMatcher(a=ref_seq, b=word_seq)
                E += res.distance()
                N += len(ref_seq)
                accuracy = (N - E) / N
                # update progress bar
                pbar.set_description(f'Test acc. {accuracy:.6f}')
                pbar.update(1)
        return accuracy

    def parallel_test(self, dataset, batch_size=50, n_jobs=20):

        test_res = {}
        E, N = 0, 0
        with tqdm(total=len(dataset), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            for x, y, text, y_true_length, filenames in dataset.generator(batch_size=batch_size, return_filename=True):
                posteriors = self.features_to_posteriors(x)

                with multiprocessing.Pool(n_jobs) as p:
                    res = p.map(parallel_decoding, zip(posteriors, y_true_length, text, [self.hmm] * len(text)))

                for (pred_word_seq, best_path, pstar, dst), label_word_seq, post, filename \
                        in zip(res, text, posteriors, filenames):
                    ref_seq_len = len(label_word_seq.split(" "))
                    E += dst
                    N += ref_seq_len
                    accuracy = (N - E) / N

                    # update progress bar
                    pbar.set_description(f'Test acc. {accuracy:.6f}')
                    pbar.update(1)

                    test_res[filename] = {'pred_word_seq': ' '.join(pred_word_seq), 'pstar': pstar,
                                          'best_path': best_path.tolist(), 'label_word_seq': label_word_seq,
                                          # 'posteriors': post.tolist(), takes so much space!!
                                          }

        return accuracy, test_res

    def features_to_posteriors(self, features, is_training=False):
        """
        Calculates posteriors for audio file.

        :param audio_file: *.wav file
        :param parameters: parameters for feature extraction

        :return: posteriors
        """
        # x = torch.from_numpy(features).float().cuda()

        y = self.forward(features, is_training=is_training)
        y = F.softmax(y, dim=2).cpu().data.numpy().squeeze()

        return y

    def wav_to_posteriors(self, audio_file):
        """
        Calculates posteriors for audio file.

        :param audio_file: *.wav file
        :param parameters: parameters for feature extraction

        :return: posteriors
        """
        x, _ = torchaudio.load(audio_file)

        # x = x.data.numpy()
        return self.features_to_posteriors(x).squeeze()

    def compute_loss_batch(self, batch_x, batch_ref_label, batch_ref_label_true_lengths, device='cuda',
                           important_indices=None, is_training=False):
        assert len(batch_x) == len(batch_ref_label)

        batch_y = self.forward(batch_x, is_training=is_training)

        # sometimes the targets dimensions differ (by one frame)
        if batch_ref_label.shape[1] != batch_y.shape[1]:
            diff = batch_y.shape[1] - batch_ref_label.shape[1]
            batch_y = batch_y[:, :batch_ref_label.shape[1], :]

            if diff > 1:
                raise ValueError('Frame difference larger than 1!')

        batch_ref_label = torch.argmax(batch_ref_label, dim=2)

        loss_all = torch.nn.functional.cross_entropy(batch_y.permute(0, 2, 1), batch_ref_label, reduction='none')

        # loss_mask determines the classification of which phonemes are important. i.e., padded phonemes are not important!
        loss_mask = torch.zeros(loss_all.shape).to(device)
        for i, true_length in enumerate(batch_ref_label_true_lengths):
            loss_mask[i, :true_length] = 1.0 / true_length

        # Note we don't need to do average here. We already did that by setting the loss_mask values to 1.0 / true_length
        effective_loss = (loss_all * loss_mask).sum(dim=1)

        # Now, let's compute the average over the number of samples in the batch
        effective_loss_mean = effective_loss.mean()

        if important_indices:
            assert len(important_indices) == len(batch_x)
            loss_mask = torch.zeros(loss_all.shape).to(device)
            for i, frame_indices in enumerate(important_indices):
                loss_mask[i, frame_indices] = 1.0 / len(frame_indices)

            effective_imp_loss = (loss_all * loss_mask).sum(dim=1)

            effective_imp_loss_mean = effective_imp_loss.mean()
            return effective_loss_mean, effective_imp_loss_mean

        else:
            return effective_loss_mean

    def compute_loss_single(self, x, ref_label, important_indices=None, is_training=False):
        y = self.forward(x, is_training=is_training)
        y = y.squeeze()  # since it's a single sample (not a batch), we remove the dimension batch.

        # sometimes the targets dimensions differ (by one frame)
        if ref_label.shape[0] != y.shape[0]:
            diff = y.shape[0] - ref_label.shape[0]
            if diff == 1:
                y = y[:ref_label.shape[0], :]
            elif diff == -1:  ## Added for multi target generated targets!!!!
                ref_label = ref_label[:y.shape[0]]

            if diff > 1:
                raise ValueError('Frame difference larger than 1!')

        if important_indices:
            y = y[important_indices]
            ref_label = ref_label[important_indices]

        if ref_label.dim() == 2:
            ref_label = torch.argmax(ref_label, dim=1)
        else:
            assert ref_label.dim() == 1

        return torch.nn.functional.cross_entropy(y, ref_label)


def roll(x, n):
    return torch.cat((x[:, -n:, :], x[:, :-n, :]), dim=1)


def add_context_pytorch(feats, left_context=4, right_context=4):
    """
    Adds context to the features.
    :param feats: extracted features.
    :param left_context: Number of predecessors.
    :param right_context: Number of succcessors.
    :return: Features with context.
    """

    feats_context = feats.unsqueeze(3)
    for i in range(1, left_context + 1):
        tmp = roll(feats, i).unsqueeze(3)
        feats_context = torch.cat((tmp, feats_context), 3)

    for i in range(1, right_context + 1):
        tmp = roll(feats, -i).unsqueeze(3)
        feats_context = torch.cat((feats_context, tmp), 3)

    return feats_context


class TwoLayerLight(BaseModel):
    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(TwoLayerLight, self).__init__(feature_parameters, hmm, dropout=dropout,
                                            test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 100)
        self.fch1h2 = nn.Linear(100, 100)
        self.fch2y = nn.Linear(100, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)
        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)

        out = h2 if penu else self.fch2y(h2)

        if return_spectrogram:
            return spect, out
        else:
            return out


class TwoLayerPlus(BaseModel):

    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(TwoLayerPlus, self).__init__(feature_parameters, hmm, dropout=dropout,
                                           test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 200)
        self.fch1h2 = nn.Linear(200, 100)
        self.fch2y = nn.Linear(100, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)

        if self.dropout > 0.0 and (is_training or self.test_dropout_enabled):
            h1 = F.dropout(h1, p=self.dropout)

        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)

        out = h2 if penu else self.fch2y(h2)

        if return_spectrogram:
            return spect, out
        else:
            return out


class ThreeLayer(BaseModel):
    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(ThreeLayer, self).__init__(feature_parameters, hmm, dropout=dropout,
                                         test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 100)
        self.fch1h2 = nn.Linear(100, 100)
        self.fch1h3 = nn.Linear(100, 100)
        self.fch2y = nn.Linear(100, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)
        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)
        h3 = self.fch1h3(h2)
        h3 = F.relu(h3)

        out = h3 if penu else self.fch2y(h3)

        if return_spectrogram:
            return spect, out
        else:
            return out


class ThreeLayerPlus(BaseModel):
    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(ThreeLayerPlus, self).__init__(feature_parameters, hmm, dropout=dropout,
                                             test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 200)
        self.fch1h2 = nn.Linear(200, 150)
        self.fch1h3 = nn.Linear(150, 100)
        self.fch2y = nn.Linear(100, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)
        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)
        h3 = self.fch1h3(h2)
        h3 = F.relu(h3)

        out = h3 if penu else self.fch2y(h3)

        if return_spectrogram:
            return spect, out
        else:
            return out

class FourLayerPlus(BaseModel):
    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(FourLayerPlus, self).__init__(feature_parameters, hmm, dropout=dropout,
                                             test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 300)
        self.fch1h2 = nn.Linear(300, 200)
        self.fch1h2_2 = nn.Linear(200, 200)
        self.fch1h3 = nn.Linear(200, 200)
        self.fch2y = nn.Linear(200, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)
        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)
        h2_2 = self.fch1h2_2(h2)
        h2_2 = F.relu(h2_2)
        h3 = self.fch1h3(h2_2)
        h3 = F.relu(h3)

        out = h3 if penu else self.fch2y(h3)

        if return_spectrogram:
            return spect, out
        else:
            return out


class ThreeLayerPlusPlus(BaseModel):
    def __init__(self, feature_parameters, hmm, dropout=0.0, test_dropout_enabled=False):
        super(ThreeLayerPlusPlus, self).__init__(feature_parameters, hmm, dropout=dropout,
                                             test_dropout_enabled=test_dropout_enabled)

        self.fcxh1 = nn.Linear(3 * self.n_mfcc * (self.left_context + self.right_context + 1), 400)
        self.fch1h2 = nn.Linear(400, 300)
        self.fch1h3 = nn.Linear(300, 200)
        self.fch2y = nn.Linear(200, self.hmm.get_num_states())

    def forward(self, x, penu=False, is_training=False, return_spectrogram=False):
        # normalize input based on the maximum value across the dataset!
        # x = x / 0.5648193359375  # torch.max(torch.abs(x)) # , dim=1)[0].reshape(-1, 1)

        # calc mfcc
        spect, mfcc = self.mfcc(x)

        # feature extraction is adding an (unnecessary) additional frame
        # if input is multiple of frame size
        # => drop this so input length == output length
        # if x.shape[1] % self.hop_size_samples == 0:
        #     mfcc = mfcc[:,:,:-1]

        # add delta and delta deltas
        deltas = self.deltas(mfcc)
        deltadeltas = self.deltas(deltas)
        mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

        mfcc = mfcc.permute(0, 2, 1)

        mfcc_context = add_context_pytorch(mfcc, self.left_context, self.right_context)

        h1 = self.fcxh1(torch.flatten(mfcc_context, start_dim=2))
        h1 = F.relu(h1)
        h2 = self.fch1h2(h1)
        h2 = F.relu(h2)
        h3 = self.fch1h3(h2)
        h3 = F.relu(h3)

        out = h3 if penu else self.fch2y(h3)

        if return_spectrogram:
            return spect, out
        else:
            return out


def init_model(model_type, feature_parameters, hmm, device='cuda', dropout=0.0, test_dropout_enabled=False):
    return eval(model_type)(feature_parameters, hmm, dropout, test_dropout_enabled).to(device)
