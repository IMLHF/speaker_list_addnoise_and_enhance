import os
import soundfile
from decoder import build_session
from FLAGS import PARAM
from utils import audio_tool
from utils import spectrum_tool
import numpy as np
import time
import sys
import multiprocessing
import copy


snr_min = 0
snr_max = 25

num_process = 16

# local
# root_dir = 'test_dir'
# new_root_dir = 'test_dir'

# 15047
root_dir = "/home/root1/aishell/iOS/data"
new_root_dir = "/home/root1/aishell/iOS/data"

# 15123
# root_dir = "/data/datalhf"
# new_root_dir = "/fast/aishell2_addnoise_enhanced"

clean_speakers_dir = os.path.join(root_dir, "wav")
addnoise_dir_name = "wav_addnoise_snr0_25"
enhanced_dir_name = "wav_enhanced_snr0_25"
# addnoise_speakers_dir = os.path.join(root_dir, "wav_addnoise_snr-5+5")
# enhanced_speakers_dir = os.path.join(root_dir, "wav_enhanced")

def _addnoise_and_decoder_one_batch(i_p, speaker_id, sub_process_speaker_num, waves_dir, noise_dir, sess, model):
  """
  x_wav, y_wav_est
  """
  s_time = time.time()
  noise_dir_list = [os.path.join(noise_dir, _dir) for _dir in os.listdir(noise_dir)]
  n_noise = len(noise_dir_list)
  wave_dir_list = [os.path.join(waves_dir, _dir) for _dir in os.listdir(waves_dir)]

  # print(len(wave_dir_list), os.path.dirname(wave_dir_list[0]))

  # mix && get input
  # x_batch = [] # [n_wav, time, 257]
  # x_theta_batch = [] # [n_wav, time, 257]
  # x_lengths = [] # [n_wav]
  batch_size = 0
  for wav_dir in wave_dir_list:
    batch_size += 1
    y_wave, sr_y = audio_tool.read_audio(wav_dir)
    noise_id = np.random.randint(n_noise)
    noise_wave, sr_n = audio_tool.read_audio(noise_dir_list[noise_id])
    noise_wave = audio_tool.repeat_to_len(noise_wave, len(y_wave))
    x_wave, alpha = audio_tool._mix_wav_by_randomSNR(y_wave, noise_wave)

    assert sr_y == sr_n and sr_y == 16000, 'sr error sr_y:%d, sr_n %d' % (sr_y, sr_n)
    x_wav_dir = wav_dir.replace('wav', addnoise_dir_name, 1)
    x_wav_dir = x_wav_dir.replace(root_dir, new_root_dir, 1)
    x_wav_father_dir = os.path.dirname(x_wav_dir)
    if not os.path.exists(x_wav_father_dir):
      os.makedirs(x_wav_father_dir)
    audio_tool.write_audio(x_wav_dir, x_wave, sr_y)

    x_spec_t = spectrum_tool.magnitude_spectrum_librosa_stft(x_wave, # [time, 257]
                                                             PARAM.NFFT,
                                                             PARAM.OVERLAP)
    x_phase_t = spectrum_tool.phase_spectrum_librosa_stft(x_wave,
                                                          PARAM.NFFT,
                                                          PARAM.OVERLAP)
    # x_batch.append(x_spec_t)
    # x_theta_batch.append(x_phase_t)
    # x_lengths.append(np.shape(x_spec_t)[0])

    x_batch = np.array([x_spec_t], dtype=np.float32)
    x_theta_batch = np.array([x_phase_t], dtype=np.float32)
    x_lengths = np.array([np.shape(x_spec_t)[0]], dtype=np.int32)

    # enhance
    y_mag_est = sess.run(
        model.y_mag_estimation,
        feed_dict={
            model.x_mag: x_batch,
            model.x_theta: x_theta_batch,
            model.lengths: x_lengths,
        })

    # istf && save
    if PARAM.RESTORE_PHASE != 'MIXED':
      raise ValueError('Please set PARAM.RESTORE_PHASE=MIXED.')
    # istft
    y_mag_est = y_mag_est*np.exp(1j*x_phase_t)
    reY = spectrum_tool.librosa_istft(y_mag_est, PARAM.NFFT, PARAM.OVERLAP)
    y_wav_dir = wav_dir.replace('wav', enhanced_dir_name, 1)
    y_wav_dir = y_wav_dir.replace(root_dir, new_root_dir, 1)
    y_wav_father_dir = os.path.dirname(y_wav_dir)
    if not os.path.exists(y_wav_father_dir):
      os.makedirs(y_wav_father_dir)
    audio_tool.write_audio(y_wav_dir, reY, PARAM.FS)

  max_len = np.max(x_lengths)

  e_time = time.time()
  print("\n----------------\n"
        "%d workers\n"
        "%s\n"
        "Worker_id %03d, rate of progress: %d/%d\n"
        "time_step_max_len: %d\n"
        "batch_sie: %d\n"
        'batch_cost_time: %ds\n' % (
            num_process,
            time.ctime(), i_p+1,
            speaker_id, sub_process_speaker_num,
            max_len, batch_size, e_time-s_time),
        flush=True)


def min_process(i_p,  clean_speaker_dir_list, s_site, e_site, noise_dir):

  # n_gpu = 1
  os.environ['CUDA_VISIBLE_DEVICES'] = "" #str(i_p%n_gpu)
  session, model = build_session(PARAM.CHECK_POINT, None)

  speaker_id = 0
  sub_process_speaker_num = e_site - s_site

  for clean_speaker_dir in clean_speaker_dir_list[s_site:e_site]:
    # (wav/S0001, session, model)
    speaker_id += 1
    if not os.path.isdir(clean_speaker_dir):
      continue
    _addnoise_and_decoder_one_batch(i_p, speaker_id, sub_process_speaker_num, clean_speaker_dir, noise_dir, session, model)

if __name__ == "__main__":


  noise_dir = os.path.join(root_dir,'noise')

  clean_speaker_dir_list = os.listdir(clean_speakers_dir)
  clean_speaker_dir_list = [os.path.join(clean_speakers_dir, _dir) for _dir in clean_speaker_dir_list]
  # print(clean_speaker_dir_list)
  print("speaker_num:", len(clean_speaker_dir_list), flush=True)

  num_speakers = len(clean_speaker_dir_list)
  n_speaker_per_process = int(num_speakers/num_process)
  n_speaker_per_process = 1 if n_speaker_per_process == 0 else n_speaker_per_process

  pool = multiprocessing.Pool(num_process)
  s_time = time.time()
  for i in range(num_process):
    s_site = i*n_speaker_per_process
    e_site = s_site + n_speaker_per_process
    if i == (num_process-1):
      e_site = num_speakers
    if s_site >= num_speakers:
      break
    # print(s_site, e_site)
    # min_process(i, clean_speaker_dir_list, s_site, e_site, noise_dir)
    pool.apply_async(min_process, (i, clean_speaker_dir_list,
                                   s_site, e_site, noise_dir))

  pool.close()
  pool.join()
  e_time = time.time()
  print("Total cost time %ds" % (e_time-s_time))
  # OMP_NUM_THREADS=1




