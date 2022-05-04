# coding: utf-8
"""
Module to represents whole models
"""
from typing import Callable
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import ConfigurationError, log_peakiness, join_strings

from joeynmt.metrics import bleu

logger = logging.getLogger(__name__)

class RewardRegressionModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.l1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.l2=nn.Linear(H, D_out)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X)))

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super().__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self._loss_function = None # set by the TrainManager

    @property
    def loss_function(self):
        return self._x

    @loss_function.setter
    def loss_function(self, loss_function: Callable):
        self._loss_function = loss_function

    def reinforce(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
            src_length: Tensor, temperature: float, topk: int, log_probabilities: False, pickle_logs:False):

        """ Computes forward pass for Policy Gradient aka REINFORCE
        
        Encodes source, then step by step decodes and samples token from output distribution.
        Calls the loss function to compute the BLEU and loss
        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :return: loss, logs
        """

        encoder_output, encoder_hidden = self._encode(src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        distributions = []
        log_probs = 0
        # init hidden state in case of using rnn decoder  
        hidden = self.decoder._init_hidden(encoder_hidden) \
            if hasattr(self.decoder,'_init_hidden') else 0
        attention_vectors = None
        finished = src_mask.new_zeros((batch_size)).byte()
        # decode tokens
        for _ in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            next_word = distrib.sample()
            log_probs += distrib.log_prob(next_word)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(next_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size:
                    break
        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # get reinforce loss
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings, log_probs)
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, distributions,
        trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)) \
        if log_probabilities else (batch_loss, [])

    def forward(self, return_type: str = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """ Interface for multi-gpu

        For DataParallel, We need to encapsulate all model call: model.encode(),
        model.decode(), and model.encode_decode() by model.__call__().
        model.__call__() triggers model.forward() together with pre hooks
        and post hooks, which take care of multi-gpu distribution.

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        return_tuple = (None, None, None, None)
        if return_type == "loss":
            assert self.loss_function is not None

            out, _, _, _ = self._encode_decode(**kwargs)

            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])

            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, None, None, None)

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(**kwargs)

            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(**kwargs)

            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)

        elif return_type == "reinforce":
            loss, logging = self.reinforce(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        return return_tuple

    # pylint: disable=arguments-differ
    def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                       src_length: Tensor, trg_mask: Tensor = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self._encode(src=src,
                                                      src_length=src_length,
                                                      src_mask=src_mask,
                                                      **kwargs)

        unroll_steps = trg_input.size(1)
        assert "decoder_hidden" not in kwargs
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask, **kwargs)

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor,
                **_kwargs) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask,
                            **_kwargs)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None, **_kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask,
                            **_kwargs)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return f"{self.__class__.__name__}(\n\tencoder={self.encoder}," \
                f"\n\tdecoder={self.decoder},\n\tsrc_embed={self.src_embed}," \
                f"\n\ttrg_embed={self.trg_embed})"


class _DataParallel(nn.DataParallel):
    """ DataParallel wrapper to pass through the model attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    logger.info("Building an encoder-decoder model...")
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    # initialize embeddings from file
    pretrained_enc_embed_path = cfg["encoder"]["embeddings"].get(
        "load_pretrained", None)
    pretrained_dec_embed_path = cfg["decoder"]["embeddings"].get(
        "load_pretrained", None)
    if pretrained_enc_embed_path:
        logger.info("Loading pretraind src embeddings...")
        model.src_embed.load_from_file(pretrained_enc_embed_path, src_vocab)
    if pretrained_dec_embed_path and not cfg.get("tied_embeddings", False):
        logger.info("Loading pretraind trg embeddings...")
        model.trg_embed.load_from_file(pretrained_dec_embed_path, trg_vocab)

    logger.info("Enc-dec model built.")
    return model
