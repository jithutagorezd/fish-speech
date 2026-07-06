from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, DAC):
            # For large sequences, decode in chunks to save memory
            codes_shape = codes.shape
            if codes_shape[-1] > 512:  # Large token sequence
                logger.info(f"Large sequence detected ({codes_shape[-1]} tokens), using chunked decoding...")
                chunk_size = 256
                decoded_chunks = []
                
                for i in range(0, codes_shape[-1], chunk_size):
                    chunk = codes[:, i:i+chunk_size]
                    logger.info(f"Decoding chunk {i//chunk_size + 1}: tokens [{i}:{i+chunk_size}]")
                    
                    with torch.no_grad():
                        decoded_chunk = self.decoder_model.from_indices(chunk[None])[0]
                    decoded_chunks.append(decoded_chunk)
                    
                    # Clear memory between chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate chunks
                result = torch.cat(decoded_chunks, dim=-1).squeeze()
                return result
            else:
                # Small sequence, decode normally
                return self.decoder_model.from_indices(codes[None])[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(self.decoder_model, DAC):
                prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
