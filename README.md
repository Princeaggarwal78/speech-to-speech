APPROACH - 1

1. Convert source and target audio to numerical representations (e.g., mel spectrograms).
2. Combine these representations into a CSV file , FOR INPUT AND OUTPUT COMPARISONS USING LSTMS (TIME ZONE AND FREQUENCY)
3. WE WILL TRAIN THE MODEL FOR THE SAME TO PREDICT THE OTHER LANGUAGE 


APPROACH -2

1.SPEECH TO TEXT THEN TEXT TO SPEECH

APPROACH -3 
1. USING DIRECT APIS 

APPROACH -4:
This architecture eliminates the need for text data. It consists of three parts:

Encoder: Converts audio recordings into a sequence of discrete units.
Language Model: Generates new sequences of these units based on learned patterns.
Decoder: Translates the generated sequences back into speech.


Fairseq(-py) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.





1. DATASET - "https://huggingface.co/datasets/google/fleurs" 
  we extracted parallel data of english and hindi from above link which contains around 55000 audios 
##from datasets import load_dataset
##fleurs = load_dataset("google/fleurs", "hi_in", split="train").
data can be accessed from above code.

2. Data preprocessing - 

Target speech
(optional) To prepare S2S data from a speech-to-text translation (ST) dataset, see fairseq-S^2 for pre-trained TTS models and instructions on how to train and decode TTS models.
Prepare two folders, $SRC_AUDIO and $TGT_AUDIO, with ${SPLIT}/${SAMPLE_ID}.wav for source and target speech under each folder, separately. Note that for S2UT experiments, target audio sampling rate should be in 16,000 Hz, and for S2SPECT experiments, target audio sampling rate is recommended to be in 22,050 Hz.
To prepare target discrete units for S2UT model training, see Generative Spoken Language Modeling (speech2unit) for pre-trained k-means models, checkpoints, and instructions on how to decode units from speech. Set the output target unit files (--out_quantized_file_path) as ${TGT_AUDIO}/${SPLIT}.txt. In Lee et al. 2021, we use 100 units from the sixth layer (--layer 6) of the HuBERT Base model.
Speech-to-speech data

S2UT

Set --reduce-unit for training S2UT reduced model
Pre-trained vocoder and config ($VOCODER_CKPT, $VOCODER_CFG) can be downloaded from the Pretrained Models section. They are not required if --eval-inference is not going to be set during model training.  
Multitask data

For each multitask $TASK_NAME, prepare ${DATA_ROOT}/${TASK_NAME}/${SPLIT}.tsv files for each split following the format below: (Two tab separated columns. The sample_ids should match with the sample_ids for the speech-to-speech data in ${DATA_ROOT}/${SPLIT}.tsv.)
id  tgt_text
sample_id_0 token1 token2 token3 ...
sample_id_1 token1 token2 token3 ...
...
For each multitask $TASK_NAME, prepare ${DATA_ROOT}/${TASK_NAME}/dict.txt, a dictionary in fairseq format with all tokens for the targets for $TASK_NAME.
Create config_multitask.yaml. Below is an example of the config used for S2UT reduced with Fisher experiments including two encoder multitasks (source_letter, target_letter) and one decoder CTC task (decoder_target_ctc).

source_letter:  # $TASK_NAME
   decoder_type: transformer
   dict: ${DATA_ROOT}/source_letter/dict.txt
   data: ${DATA_ROOT}/source_letter
   encoder_layer: 6
   loss_weight: 8.0
target_letter:
   decoder_type: transformer
   dict: ${DATA_ROOT}/target_letter/dict.txt
   data: ${DATA_ROOT}/target_letter
   encoder_layer: 8
   loss_weight: 8.0
decoder_target_ctc:
   decoder_type: ctc
   dict: ${DATA_ROOT}/decoder_target_ctc/dict.txt
   data: ${DATA_ROOT}/decoder_target_ctc
   decoder_layer: 3
   loss_weight: 1.6



hubert -base
K-mean model = Modified CPC + KM100
N_CLUSTERS=<number_of_clusters_used_for_kmeans>
TYPE=<one_of_logmel/cpc/hubert/w2v2>
CKPT_PATH=<path_of_pretrained_acoustic_model>
LAYER=<layer_of_acoustic_model_to_extract_features_from>
MANIFEST=<tab_separated_manifest_of_audio_files_for_training_kmeans>
KM_MODEL_PATH=<output_path_of_the_kmeans_model>

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH


    MANIFEST=<tab_separated_manifest_of_audio_files_to_quantize>
OUT_QUANTIZED_FILE=<output_quantized_audio_file_path>

python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac"


<path_of_root_directory_containing_audio_files>
<relative_path_of_audio_file_1>\t<number_of_frames_1>
<relative_path_of_audio_file_2>\t<number_of_frames_1>
...




training

Speech-to-unit translation (S2UT)

Here's an example for training Fisher S2UT models with 100 discrete units as target:
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan  \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8


  Adjust --update-freq accordingly for different #GPUs. In the above we set --update-freq 4 to simulate training with 4 GPUs.
Set --n-frames-per-step 5 to train an S2UT stacked system with reduction ratio r=5. (Use $DATA_ROOT prepared without --reduce-unit.)
(optional) one can turn on tracking MCD loss during training for checkpoint selection by setting --eval-inference --eval-args '{"beam": 1, "max_len_a": 1}' --best-checkpoint-metric mcd_loss. It is recommended to sample a smaller subset as the validation set as MCD loss computation is time-consuming.


Speech-to-spectrogram translation (S2SPECT)

Here's an example for training Fisher S2SPECT models with reduction ratio r=5:

fairseq-train $DATA_ROOT \


fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --n-frames-per-step 5 \
  --criterion speech_to_spectrogram \
  --arch s2spect_transformer_fisher --decoder-normalize-before \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --eval-inference --best-checkpoint-metric mcd_loss \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 --weight-decay 1e-6 \
  --max-update 400000 --max-tokens 80000 --max-tokens-valid 30000  --required-batch-size-multiple 1 \
  --max-target-positions 3000 --update-freq 16 \
  --seed 1 --fp16 --num-workers 8

  Adjust --update-freq accordingly for different #GPUs. In the above we set --update-freq 16 to simulate training with 16 GPUs.
We recommend turning on MCD loss during training for the best checkpoint selection.
Unit-based HiFi-GAN vocoder

The vocoder is trained with the speech-resynthesis repo. See here for instructions on how to train the unit-based HiFi-GAN vocoder with duration prediction. The same vocoder can support waveform generation for both reduced unit sequences (with --dur-prediction set during inference) and original unit sequences


To evaluate speech translation output, we first apply ASR on the speech output and then compute BLEU score betweent the ASR decoded text and the references using sacreBLEU.

En

ASR: We use the "Wav2Vec 2.0 Large (LV-60) + Self Training / 960 hours / Libri-Light + Librispeech" En ASR model open-sourced by the wav2vec project. See instructions on how to run inference with a wav2vec-based ASR model. The model is also available on Hugging Face.
Text normalization: We use the text cleaner at https://github.com/keithito/tacotron for pre-processing reference English text for ASR BLEU evaluation.




I LEARNED ABOUT THE SPEECH TO SPEECH
REFERENCE - "Textless Speech-to-Speech Translation on Real Data"
IT requires a lot of research 
i tried to install the libraries but it didnt work correctly it is giving the same error



"  Getting requirements to build editable ... error
  error: subprocess-exited-with-error

  × Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [17 lines of output]
      Traceback (most recent call last):
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 353, in <module>
          main()
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 132, in get_requires_for_build_editable
          return hook(config_settings)
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 448, in get_requires_for_build_editable
          return self.get_requires_for_build_wheel(config_settings)
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 325, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=['wheel'])
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 295, in _get_build_requires
          self.run_setup()
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 311, in run_setup
          exec(code, locals())
        File "<string>", line 246, in <module>
      OSError: [WinError 1314] A required privilege is not held by the client: '..\\examples' -> 'fairseq\\examples'
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error



which becomes a road block for i tried to fetch data of parallel english and hindi and it was successfully. link of the data set is "https://huggingface.co/datasets/google/fleurs/tree/main/data"



