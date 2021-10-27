from pathlib import Path
import numpy as np
from math import pi
from scipy.io import wavfile
from tempfile import TemporaryDirectory
from subprocess import run, DEVNULL
import shutil
import torch
import torchaudio
import torch.nn.functional as F

torch.set_num_threads(1)

LOG_LIM = 1e-10


class Psycho:

    def __init__(self, phi, scale_with_H=False):
        self.phi = phi
        self.sampling_rate = 16000
        self.win_length = 400
        self.hop_length = 200
        self.scale_with_H = scale_with_H

    @staticmethod
    def calc_thresholds(in_signal, win_length=400, hop_length=200, out_file=None, okay_to_fail=False):
        # in_file = Path(in_file)
        # if not out_file: out_file = in_file.with_suffix(".csv")
        if out_file.is_file():
            return True
        with TemporaryDirectory() as tmp_dir:
            try:
                # copy wav in tmp dir
                tmp_wav_file = Path(tmp_dir).joinpath(out_file.with_suffix('.wav').name)
                if in_signal.device != 'cpu':
                    torchaudio.save(str(tmp_wav_file), in_signal.cpu(), 16000)
                else:
                    torchaudio.save(str(tmp_wav_file), in_signal, 16000)
                # torchaudio.save(str(out_file.parent.joinpath((out_file.name + ".tmp").replace(".", "-")).with_suffix(".wav")),
                #                 in_signal, 16000)
                # shutil.copyfile(in_file, tmp_wav_file)
                # creat wav.scp
                tmp_wav_scp = Path(tmp_dir).joinpath('wav.scp')
                tmp_wav_scp.write_text(f'data {tmp_wav_file}\n')
                # get hearing threshs
                run(f"/matlab/hearing_thresholds/run_calc_threshold.sh /usr/local/MATLAB/MATLAB_Runtime/v96 {tmp_wav_scp} {win_length} {hop_length} {tmp_dir}/",
                    shell=True)
                shutil.copyfile(Path(tmp_dir).joinpath('data_dB.csv'), out_file)
                return True
            except:
                aud = torchaudio.load(str(tmp_wav_file))[0]
                assert okay_to_fail
                return False

    def get_psycho_mask(self, complex_spectrum, threshs_file):
        tmp_complex_spectrum = complex_spectrum.detach().clone()
        # Step 1: remove offset
        offset = tmp_complex_spectrum[0, :, :]
        features = tmp_complex_spectrum[1:, :, :]
        # Step 2: represent as phase and magnitude
        a_re = features[:, :, 0];
        a_re[torch.where(a_re == 0)] = 1e-20
        b_im = features[:, :, 1]
        # phase
        phase = torch.atan(b_im / a_re)
        phase[torch.where(a_re < 0)] += np.pi
        # magnitude
        magnitude = torch.sqrt(torch.square(a_re) + torch.square(b_im))
        # Step 3: get thresholds
        assert self.phi is not None
        # import thresholds
        assert threshs_file.is_file()
        # read in hearing thresholds
        thresholds = Path(threshs_file).read_text()
        thresholds = [row.split(',') for row in thresholds.split('\n')]
        # remove padded frames (copies frames at end and beginning)
        thresholds = np.array(thresholds[4:-4], dtype=float)[:, :self.hop_length]
        thresholds = torch.tensor(thresholds, dtype=torch.float32).to(complex_spectrum.device)
        thresholds = thresholds.permute((1, 0))
        # Step 4: calc mask
        m_max = magnitude.max()
        S = 20 * torch.log10(magnitude / m_max)  # magnitude in dB
        H = thresholds - 95
        # scale with phi
        H_scaled = H + self.phi
         
        # mask 
        mask = torch.ones(S.shape).to(S.device)
        mask[torch.where(S <= H_scaled)] = 0
        mask_offset = torch.ones((1, mask.shape[1])).to(S.device)
        mask = torch.cat((mask_offset, mask), dim=0)
        mask = torch.stack((mask, mask), dim=2)
        return mask

    def complex_to_mag(self, complex_spectrum):
        complex_spectrum = complex_spectrum[:, :, 0] + complex_spectrum[:, :, 1] * 1j
        return complex_spectrum.abs()

    def complex_to_phase(self, complex_spectrum):
        complex_spectrum = complex_spectrum[:, :, 0] + complex_spectrum[:, :, 1] * 1j
        return complex_spectrum.angle()

    def mag_to_db(self, magnitude):
        return 20 * torch.log10(torch.clamp(magnitude, min=LOG_LIM))

    def complex_to_db(self, complex_spectrum):
        return self.mag_to_db(self.complex_to_mag(complex_spectrum))

    def read_threshold(self, threshs_file):
        # import thresholds
        assert threshs_file.is_file()
        # read in hearing thresholds
        thresholds = Path(threshs_file).read_text()
        thresholds = [row.split(',') for row in thresholds.split('\n')]
        # remove padded frames (copies frames at end and beginning)
        thresholds = np.array(thresholds[4:-4], dtype=float)[:, :self.hop_length]
        thresholds = torch.tensor(thresholds, dtype=torch.float32)
        thresholds = thresholds.permute((1, 0))
        # remove threshold offset and apply phi
        H = thresholds - 95 + self.phi

        return H

    def filter_imperceptible(self, signal, threshs_file):

        if self.phi is None:
            return signal

        # fft
        complex_spectrum = torch.stft(signal,
                                      n_fft=self.win_length,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=torch.hamming_window(self.win_length).to(signal.device),
                                      pad_mode='constant',
                                      onesided=True)

        # mask signal with psychoacoustic thresholds
        mask = self.get_psycho_mask(complex_spectrum, threshs_file)
        complex_spectrum_masked = complex_spectrum * mask

        # ifft
        signal_out = torch.istft(complex_spectrum_masked,
                                 n_fft=self.win_length,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=torch.hamming_window(self.win_length).to(signal.device),
                                 onesided=True)

        return signal_out

    # def forward(self, signal, signal_modified, threshs_file):
    #
    #     if self.phi is None:
    #         return signal
    #
    #     # fft original signal
    #     complex_spectrum = torch.stft(signal,
    #                                   n_fft=self.win_length,
    #                                   hop_length=self.hop_length,
    #                                   win_length=self.win_length,
    #                                   window=torch.hamming_window(self.win_length).to(signal.device),
    #                                   pad_mode='reflect',
    #                                   onesided=True,
    #                                   return_complex=True)
    #
    #     # fft modified signal
    #     complex_spectrum_modified = torch.stft(signal_modified,
    #                                            n_fft=self.win_length,
    #                                            hop_length=self.hop_length,
    #                                            win_length=self.win_length,
    #                                            window=torch.hamming_window(self.win_length).to(signal_modified.device),
    #                                            pad_mode='reflect',
    #                                            onesided=True,
    #                                            return_complex=False)
    #
    #     # get signal difference via 20*log10(|M-S|) - 20*log10(max(S))
    #     max_val_dB = torch.max(self.complex_to_db(complex_spectrum))
    #     magnitude_diff = self.complex_to_db(complex_spectrum_modified - complex_spectrum)
    #     magnitude_diff = magnitude_diff - max_val_dB
    #
    #     # import thresholds
    #     assert threshs_file.is_file()
    #     # read in hearing thresholds
    #     thresholds = Path(threshs_file).read_text()
    #     thresholds = [row.split(',') for row in thresholds.split('\n')]
    #     # remove padded frames (copies frames at end and beginning)
    #     thresholds = np.array(thresholds[4:-4], dtype=float)[:, :self.hop_length]
    #     thresholds = torch.tensor(thresholds, dtype=torch.float32).to(signal.device)
    #     thresholds = thresholds.permute((1, 0))
    #     # remove threshold offset and apply phi
    #     H = thresholds - 95 + self.phi
    #     # padd first entry (signal offset)
    #     mask_offset = torch.ones((1, H.shape[1])).to(signal.device)
    #     H = torch.cat((mask_offset, H), dim=0)
    #
    #     if H.shape[1] != magnitude_diff.shape[1]:
    #         diff = magnitude_diff.shape[1] - H.shape[1]
    #         if diff > 1:
    #             raise ValueError('Frame difference larger than 1!')
    #         magnitude_diff = magnitude_diff[:, :H.shape[1]]
    #
    #     # caclualte mask of audible ranges
    #     magnitude_diff = magnitude_diff - H
    #
    #     magnitude_clamp = torch.clamp(magnitude_diff, min=None, max=0)
    #     mask = magnitude_clamp - magnitude_diff
    #
    #     # apply mask to modified signal
    #     maginitude_modified = self.complex_to_db(complex_spectrum_modified) + mask
    #
    #     # rebuild signal with original phase
    #     phase_modified = self.complex_to_phase(complex_spectrum_modified)
    #     complex_spectrum_masked = 10 ** (maginitude_modified / 20) * torch.exp(1j * phase_modified.cpu()).to(signal.device)
    #     complex_spectrum_masked = torch.cat(
    #         (complex_spectrum_masked.real.unsqueeze(2), complex_spectrum_masked.imag.unsqueeze(2)), dim=2)
    #
    #     # ifft
    #     signal_out = torch.istft(complex_spectrum_masked,
    #                              n_fft=self.win_length,
    #                              hop_length=self.hop_length,
    #                              win_length=self.win_length,
    #                              window=torch.hamming_window(self.win_length).to(signal.device),
    #                              onesided=True)
    #
    #     return signal_out

    def scale_grads(self, signal, signal_modified, threshs_file):

        # fft original signal
        complex_spectrum = torch.stft(signal,
                                      n_fft=self.win_length,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=torch.hamming_window(self.win_length).to(signal.device),
                                      pad_mode='constant',
                                      onesided=True)

        # fft modified signal
        complex_spectrum_modified = torch.stft(signal_modified,
                                               n_fft=self.win_length,
                                               hop_length=self.hop_length,
                                               win_length=self.win_length,
                                               window=torch.hamming_window(self.win_length).to(signal_modified.device),
                                               pad_mode='constant',
                                               onesided=True)

        # get signal difference via 20*log10(|M-S|) - 20*log10(max(S))
        max_val_dB = torch.max(self.complex_to_db(complex_spectrum))
        magnitude_diff = self.complex_to_db(complex_spectrum_modified - complex_spectrum)
        magnitude_diff = magnitude_diff - max_val_dB

        # import thresholds
        assert threshs_file.is_file()
        # read in hearing thresholds
        thresholds = Path(threshs_file).read_text()
        thresholds = [row.split(',') for row in thresholds.split('\n')]
        # remove padded frames (copies frames at end and beginning)
        thresholds = np.array(thresholds[4:-4], dtype=float)[:, :self.hop_length]
        thresholds = torch.tensor(thresholds, dtype=torch.float32).to(signal.device)
        thresholds = thresholds.permute((1, 0))
        # remove threshold offset and apply phi
        H = thresholds - 95 + self.phi
        # padd first entry (signal offset)
        mask_offset = torch.ones((1, H.shape[1])).to(signal.device)
        H = torch.cat((mask_offset, H), dim=0)

        if H.shape[1] != magnitude_diff.shape[1]:
            diff = magnitude_diff.shape[1] - H.shape[1]
            if diff > 1:
                raise ValueError('Frame difference larger than 1!')
            magnitude_diff = magnitude_diff[:, :H.shape[1]]

        # caclualte mask of audible ranges
        magnitude_diff = H - magnitude_diff

        magnitude_clamp = torch.clamp(magnitude_diff, min=0, max=None)

        magnitude_clamp_min = magnitude_clamp.min()
        magnitude_clamp_max = magnitude_clamp.max()

        scale_grads = (magnitude_clamp - magnitude_clamp_min) / (magnitude_clamp_max - magnitude_clamp_min)

        if self.scale_with_H:
            H_min = H.min()
            H_max = H.max()

            H_normalized = (H - H_min) / (H_max - H_min)

            return scale_grads * H_normalized
        else:
            return scale_grads

    # def forward(self, signal, threshs_file):

    #     if self.phi is None:
    #         return signal

    #     # fft
    #     complex_spectrum = torch.stft(signal,
    #                         n_fft=self.win_length,
    #                         hop_length=self.hop_length,
    #                         win_length=self.win_length,
    #                         window=torch.hamming_window(self.win_length),
    #                         pad_mode='reflect',
    #                         onesided=True)

    #     # mask signal with psychoacoustic thresholds
    #     mask = self.get_psycho_mask(complex_spectrum, threshs_file)
    #     complex_spectrum_masked = complex_spectrum * mask

    #     # ifft
    #     signal_out = torch.istft(complex_spectrum_masked,
    #                 n_fft=self.win_length,
    #                 hop_length=self.hop_length,
    #                 win_length=self.win_length,
    #                 window=torch.hamming_window(self.win_length),
    #                 onesided=True)

    #     return signal_out

    # def convert_wav(self, in_file, poison_file, threshs_file, out_file, device='cpu'):
    #     torch_signal, _ = torchaudio.load(in_file)
    #     torch_signal = (torch.round(torch_signal * 32767)).squeeze().to(device)
    #     torch_signal_modified, _ = torchaudio.load(poison_file)
    #     torch_signal_modified = (torch.round(torch_signal_modified * 32767)).squeeze().to(device)
    #
    #     # signal_out = self.forward(torch_signal, threshs_file)
    #     signal_out = self.forward(torch_signal, torch_signal_modified, threshs_file)
    #
    #     signal_out = torch.round(signal_out).cpu().detach().numpy().astype('int16')
    #     wavfile.write(out_file, self.sampling_rate, signal_out)

    # def convert_wav(self, orig_signal, modified_signal, threshs_file, device='cpu'):
    #     orig_signal = (torch.round(orig_signal * 32767)).squeeze().to(device)
    #     modified_signal = (torch.round(modified_signal * 32767)).squeeze().to(device)
    #
    #     out_signal = self.forward(orig_signal, modified_signal, threshs_file)
    #
    #     out_signal = out_signal / 32767
    #
    #     return out_signal


# def read_audio(filename):
#     torch_signal = torch.load(filename, map_location=torch.device('cpu')).squeeze()
#     return torch_signal
#
# def read_torchaudio(filename):
#     torch_signal = torchaudio.load(filename)[0].squeeze()
#     return torch_signal
#
#
# if __name__ == "__main__":
#     # margin for hearing threshs in dB
#     # => for phi = 0, we use the original hearing threshold,
#     #    for higher values we use a more aggressive filtering,
#     #    and for smaller values we retain more from the original signal
#     import sys
#     phi = int(sys.argv[1])
#
#     # input
#     original_file = Path("/asr-python/psycho_test/TRAIN-MAN-EH-712A.pt")
#     poison_file = Path("/asr-python/psycho_test/TRAIN-MAN-EH-712A.poison.wav")
#
#     # output
#     out_file = Path(f"/asr-python/psycho_test/{poison_file.stem}.psycho-PHI{phi}.wav")
#
#     original_signal = read_audio(original_file)
#     poison_signal = read_torchaudio(poison_file)
#
#     psycho = Psycho(phi)
#     # compute hearing thresholdsd
#     # => this is relatively expensive to compute, so we cache the
#     #    threshs alongside the original wav as a .csv
#     # => Note: each call spawns its own matlab process
#     #          so multiprocessing with python should be
#     #          straightforward
#
#     threshs_file = original_file.with_suffix(".csv")
#     Psycho.calc_thresholds(original_signal, out_file=threshs_file)
#
#     print(f"[+] Remove audible parts for {poison_file.name} @ {out_file}")
#
#     filtered_signal = Psycho(phi).convert_wav(original_signal, poison_signal, threshs_file)
#
#     torchaudio.save(str(out_file), filtered_signal, 16000)

# threshs1 = Psycho.my_calc_thresholds(original_signals[0], out_file=original_files[0].with_suffix(".mycsv"))
