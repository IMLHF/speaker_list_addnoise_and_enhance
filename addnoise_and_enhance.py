import os
import soundfile
from decoder import build_session
from FLAGS import PARAM
from utils import audio_tool
from utils import spectrum_tool
import numpy as np
import time
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

snr_min = -5
snr_max = 5
speaker_n = 0
all_speaker = 1991

# root_dir = 'test_dir'
root_dir = "/home/root1/aishell/iOS/data"
clean_speakers_dir = os.path.join(root_dir, "wav")
addnoise_dir_name = "wav_addnoise_snr-5+5"
enhanced_dir_name = "wav_enhanced"
# addnoise_speakers_dir = os.path.join(root_dir, "wav_addnoise_snr-5+5")
# enhanced_speakers_dir = os.path.join(root_dir, "wav_enhanced")

def addnoise_and_decoder_one_batch(waves_dir, noise_dir, sess, model):
  """
  x_wav, y_wav_est
  """
  s_time = time.time()
  global speaker_n
  speaker_n += 1
  print("\n----------------\n","%d/%d"%(speaker_n,all_speaker))
  sys.stdout.flush()
  noise_dir_list = [os.path.join(noise_dir, _dir) for _dir in os.listdir(noise_dir)]
  n_noise = len(noise_dir_list)
  wave_dir_list = [os.path.join(waves_dir, _dir) for _dir in os.listdir(waves_dir)]

  # print(len(wave_dir_list), os.path.dirname(wave_dir_list[0]))

  # mix && get input
  x_batch = [] # [n_wav, time, 257]
  x_theta_batch = [] # [n_wav, time, 257]
  x_lengths = [] # [n_wav]
  for wav_dir in wave_dir_list:
    y_wave, sr_y = audio_tool.read_audio(wav_dir)
    noise_id = np.random.randint(n_noise)
    noise_wave, sr_n = audio_tool.read_audio(noise_dir_list[noise_id])
    noise_wave = audio_tool.repeat_to_len(noise_wave, len(y_wave))
    x_wave, alpha = audio_tool._mix_wav_by_randomSNR(y_wave, noise_wave)

    assert sr_y == sr_n and sr_y == 16000, 'sr error sr_y:%d, sr_n %d' % (sr_y, sr_n)
    x_wav_dir = wav_dir.replace('wav', addnoise_dir_name, 1)
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
    x_batch.append(x_spec_t)
    x_theta_batch.append(x_phase_t)
    x_lengths.append(np.shape(x_spec_t)[0])

  max_len = np.max(x_lengths)
  print("time_step_max_len:",max_len)
  sys.stdout.flush()

  x_batch_mat = []
  x_theta_batch_mat = []
  for x_spec, x_theta, length in zip(x_batch, x_theta_batch, x_lengths):
    x_spec_mat = np.pad(x_spec, ((0,max_len-length),(0,0)), 'constant', constant_values=((0,0),(0,0)))
    x_theta_mat = np.pad(x_theta, ((0,max_len-length),(0,0)), 'constant', constant_values=((0,0),(0,0)))
    x_batch_mat.append(x_spec_mat)
    x_theta_batch_mat.append(x_theta_mat)

  x_batch = np.array(x_batch_mat, dtype=np.float32)
  x_theta_batch = np.array(x_theta_batch_mat, dtype=np.float32)
  x_lengths = np.array(x_lengths, dtype=np.int32)


  # enhance
  y_mag_est_batch = sess.run(
      model.y_mag_estimation,
      feed_dict={
          model.x_mag: x_batch,
          model.x_theta: x_theta_batch,
          model.lengths: x_lengths,
      })

  # istf && save
  print(np.shape(y_mag_est_batch), np.shape(x_theta_batch), np.shape(x_lengths))
  sys.stdout.flush()
  for y_mag_est, x_theta, length, wav_dir in zip(y_mag_est_batch, x_theta_batch, x_lengths, wave_dir_list):
    if PARAM.RESTORE_PHASE != 'MIXED':
      raise ValueError('Please set PARAM.RESTORE_PHASE=MIXED.')
    # cat padding
    y_mag_est = y_mag_est[:length,:]
    x_theta = x_theta[:length,:]

    # istft
    y_mag_est = y_mag_est*np.exp(1j*x_theta)
    reY = spectrum_tool.librosa_istft(y_mag_est, PARAM.NFFT, PARAM.OVERLAP)
    y_wav_dir = wav_dir.replace('wav', enhanced_dir_name, 1)
    y_wav_father_dir = os.path.dirname(y_wav_dir)
    if not os.path.exists(y_wav_father_dir):
      os.makedirs(y_wav_father_dir)
    audio_tool.write_audio(y_wav_dir, reY, PARAM.FS)

  e_time = time.time()
  print('batch_cost_time: %ds' % (e_time-s_time), flush=True)

if __name__ == "__main__":

  session, model = build_session(PARAM.CHECK_POINT, None)

  noise_dir = os.path.join(root_dir,'noise')

  clean_speaker_dir_list = os.listdir(clean_speakers_dir)
  clean_speaker_dir_list = [os.path.join(clean_speakers_dir, _dir) for _dir in clean_speaker_dir_list]
  # print(clean_speaker_dir_list)
  print("speaker_num:", len(clean_speaker_dir_list), flush=True)

  for clean_speaker_dir in clean_speaker_dir_list:
    # (wav/S0001, session, model)
    if not os.path.isdir(clean_speaker_dir):
      continue
    addnoise_and_decoder_one_batch(clean_speaker_dir, noise_dir, session, model)
