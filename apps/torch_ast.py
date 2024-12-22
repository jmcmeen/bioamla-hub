from transformers import ASTFeatureExtractor,AutoModelForAudioClassification
import torch
import torchaudio
from torchaudio.transforms import Resample

def wave_file_ast_inference(wave_path : str, model_path : str, sample_rate : int):
  waveform, orig_freq = load_waveform_tensor(wave_path)
  waveform = resample_waveform_tensor(waveform, orig_freq, sample_rate)
  input_values = extract_features(waveform, sample_rate)
  model = load_pretrained_ast_model(model_path)
  return ast_predict(input_values, model)

def ast_predict(input_values, model: AutoModelForAudioClassification):
  with torch.no_grad():
    outputs = model(input_values)

  predicted_class_idx = outputs.logits.argmax(-1).item()
  return model.config.id2label[predicted_class_idx]

def load_waveform_tensor(filepath : str):
  waveform, sample_rate = torchaudio.load(filepath)
  return (waveform, sample_rate)

def resample_waveform_tensor(waveform_tensor : torch.Tensor, orig_freq : int, new_freq : int):
  resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)
  waveform_tensor = resampler(waveform_tensor)
  return waveform_tensor

def extract_features(waveform_tensor : torch.Tensor, sample_rate : int):
  waveform_tensor = waveform_tensor.squeeze().numpy()
  feature_extractor = ASTFeatureExtractor()
  inputs = feature_extractor(waveform_tensor, sampling_rate=sample_rate, padding="max_length", return_tensors="pt")
  return inputs.input_values

def load_pretrained_ast_model(model_path : str) -> AutoModelForAudioClassification:
  return AutoModelForAudioClassification.from_pretrained(model_path)