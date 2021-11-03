import os
import os.path as osp
import librosa
import numpy as np
from math import pi
from glob import glob
from tqdm import tqdm

from ambisonics.common import spherical_harmonics_matrix
from ambisonics.hrir import CIPIC_HRIR
from ambisonics.position import PositionalSource, Position

class StereoTool():
    def __init__(self, sr=16000):
        
        self.sr = sr

        # build binauralizer
        hrtf_dir = "./subject03" 
        # binauralizer = SourceBinauralizer(use_hrtfs=True, cipic_dir=hrtf_dir)
        self.hrir_db = CIPIC_HRIR(hrtf_dir)
        # encode to ambisonics, and then stereo
        speakers_phi = (2. * np.arange(2*4) / float(2*4) - 1.) * np.pi
        self.speakers_pos = [Position(phi, 0, 1, 'polar') for phi in speakers_phi]
        self.sph_mat = spherical_harmonics_matrix(self.speakers_pos, max_order=1) 

    def construct_stereo_direct(self, pst_sources):
        '''
        directly convert mono audio to binaural one with HRIR. 
        '''
        stereo = np.zeros((2, len(pst_sources[0].signal)))
        for src in pst_sources:
            left_hrir, right_hrir = self.hrir_db.get_closest(src.position)[1:]
            left_signal = np.convolve(src.signal, np.flip(left_hrir, axis=0), 'valid')
            right_signal = np.convolve(src.signal, np.flip(right_hrir, axis=0), 'valid')

            n_valid, i_start = left_signal.shape[0], left_hrir.shape[0] - 1
            stereo[0, i_start:(i_start + n_valid)] += left_signal
            stereo[1, i_start:(i_start + n_valid)] += right_signal

        return stereo

    def construct_stereo_ambi(self, pst_sources=None, ambisonic=None, HRTF=True):
        '''
        convert mono audio to ambisonics(optional)
        transfer ambisonics to a set of virtual speakers, and convert to binaural audio finally. 
        '''
        if ambisonic is None:
            assert pst_sources is not None
            ambisonic = self.get_ambi(pst_sources) 

        assert ambisonic.shape[1] == 4
        if not HRTF:
            stereo = np.stack((
                ambisonic[:, 0] / 2 + ambisonic[:, 1] / 2,
                ambisonic[:, 0] / 2 - ambisonic[:, 1] / 2
            ))
            return stereo
        array_speakers_sound = np.dot(ambisonic, self.sph_mat.T)
        array_sources = [PositionalSource(array_speakers_sound[:, i], speaker_pos, \
            self.sr) for i, speaker_pos in enumerate(self.speakers_pos)]

        return self.construct_stereo_direct(array_sources)

    def get_ambi(self, pst_sources):
        '''
        convert mono audio into ambisonics
        '''
        # encode to ambisonics
        Y = spherical_harmonics_matrix([src.position for src in pst_sources], max_order=1)
        signals = np.stack([src.signal for src in pst_sources], axis=1)
        ambisonic = np.dot(signals, Y) # shape: [Len, 4]

        return ambisonic

    def sample_pst_srcs(self, audio_file, azi=0, ele=0, r=3):
        mono, _ = librosa.load(audio_file, sr=self.sr, mono=False)
        pst_sources = [PositionalSource(mono, Position(azi, ele, r, 'polar'), self.sr)]

        return pst_sources

if __name__ == '__main__':
    tool = StereoTool()
    audio_file = 'src/0001.wav'
    azi = 60 / 180 * pi # assign the azimuth of the sound source
    ele = 30 / 180 * pi # assign the elevation of the sound source 
    pst_sources = tool.sample_pst_srcs(audio_file, azi, ele) 
    stereo = tool.construct_stereo_ambi(pst_sources)
    librosa.output.write_wav('ambi.wav', stereo, tool.sr)
