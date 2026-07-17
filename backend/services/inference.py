import os
import sys
import time
import uuid
import torch
import soundfile as sf

# Add workspace root to sys.path to import mtcnet
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Patch torch.load to set weights_only=False by default (for PyTorch 2.6+)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from mtcnet.config import MTCNetConfig
from mtcnet.model import MTCNet

class SpeakerSeparationService:
    def __init__(self, outputs_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Load weights paths
        self.weights_2spk_path = os.path.join(workspace_root, "MiniLibriMix_mtcnet_weights.pth")
        self.weights_3spk_path = os.path.join(workspace_root, "Libri3Mix_mtcnet_weights.pth")
        
        print(f"Initializing SpeakerSeparationService on device: {self.device}")
        
        # Pre-load models
        self.model_2spk = self._load_model(self.weights_2spk_path)
        self.model_3spk = self._load_model(self.weights_3spk_path)
        
        print("Models loaded successfully and ready for inference!")

    def _load_model(self, checkpoint_path: str) -> MTCNet:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load configuration
        config = checkpoint.get("config", MTCNetConfig())
        
        # Initialize model with configuration
        model = MTCNet(config)
        
        # Get state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        
        # Map state dict keys to handle MossFormerBlock inner block formatting
        mapped_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("backbone.shared_block.") and not k.startswith("backbone.shared_block.block."):
                new_k = k.replace("backbone.shared_block.", "backbone.shared_block.block.")
            mapped_state_dict[new_k] = v
            
        model.load_state_dict(mapped_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_audio(self, file_path: str) -> tuple[torch.Tensor, int, float]:
        """Loads audio, converts to mono, resamples to 8000 Hz, returns tensor, samplerate, duration."""
        wav, sr = sf.read(file_path, dtype="float32")
        duration = len(wav) / sr
        
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
            
        waveform = torch.from_numpy(wav)
        
        # Resample to 8000 Hz if necessary
        target_sr = 8000
        if sr != target_sr:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
        return waveform, target_sr, duration

    def separate(self, input_file_path: str) -> dict:
        t_start = time.time()
        
        # 1. Audio Preprocessing
        waveform_8k, sample_rate, duration = self.preprocess_audio(input_file_path)
        mixture = waveform_8k.unsqueeze(0).to(self.device)  # [1, L]
        
        t_preprocess_end = time.time()
        
        # 2. Speaker Analysis (First-pass using 3-speaker model)
        with torch.no_grad():
            analysis_outputs = self.model_3spk(mixture)
            
        presence_probs = analysis_outputs["presence_probs"][0].cpu().tolist()
        
        # Post-process count: check the energy of separated sources to filter out noise channels.
        # Since model_3spk was trained on 3-speaker data only, its query-3 presence probability
        # is often artificially high. We compute the RMS energy of each separated source.
        separated_temp = analysis_outputs["waveforms"][0]  # [N_predicted, L]
        rms_values = [torch.sqrt(torch.mean(separated_temp[i] ** 2)).item() for i in range(separated_temp.size(0))]
        sorted_rms = sorted(rms_values)
        
        if len(sorted_rms) >= 3:
            energy_ratio = sorted_rms[0] / (sorted_rms[1] + 1e-8)
            print(f"Analysis RMS: {rms_values}, Sorted: {sorted_rms}, Ratio: {energy_ratio:.4f}")
            # If the quietest channel has less than 15% of the energy of the second-quietest channel,
            # it is likely residual noise rather than an active speaker.
            if energy_ratio < 0.15:
                predicted_count = 2
            else:
                predicted_count = 3
        else:
            predicted_count = len(sorted_rms)
        
        t_analysis_end = time.time()
        
        # 3. Automatic Model Selection
        if predicted_count <= 2:
            selected_model = self.model_2spk
            model_used = "MTC-Net (2 Speaker)"
        else:
            selected_model = self.model_3spk
            model_used = "MTC-Net (3 Speaker)"
            
        # 4. Speech Separation
        t_inference_start = time.time()
        with torch.no_grad():
            outputs = selected_model(mixture)
            
        separated = outputs["waveforms"][0]  # [N_predicted, L]
        actual_count = int(outputs["predicted_count"][0].item())
        
        t_inference_end = time.time()
        
        # 5. Audio Generation (saving separate channels to unique request path)
        request_id = str(uuid.uuid4())
        request_out_dir = os.path.join(self.outputs_dir, request_id)
        os.makedirs(request_out_dir, exist_ok=True)
        
        separated_audio_files = []
        for i in range(actual_count):
            audio_np = separated[i].detach().cpu().numpy()
            filename = f"speaker_{i+1}.wav"
            full_path = os.path.join(request_out_dir, filename)
            
            # Save audio using soundfile
            sf.write(full_path, audio_np, sample_rate, subtype="FLOAT")
            
            # Save relative URL path
            separated_audio_files.append(f"/static/outputs/{request_id}/{filename}")
            
        t_end = time.time()
        
        # Prepare execution timings
        preprocessing_time = t_preprocess_end - t_start
        analysis_time = t_analysis_end - t_preprocess_end
        inference_time = t_inference_end - t_inference_start
        generation_time = t_end - t_inference_end
        total_time = t_end - t_start
        
        return {
            "detected_speakers": actual_count,
            "model_used": model_used,
            "processing_time": total_time,
            "inference_time": inference_time,
            "timings": {
                "preprocessing": preprocessing_time,
                "analysis": analysis_time,
                "inference": inference_time,
                "generation": generation_time,
                "total": total_time
            },
            "separated_audio_files": separated_audio_files,
            "original_duration": duration,
            "presence_probs": presence_probs
        }
