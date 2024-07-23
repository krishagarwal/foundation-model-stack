# ./fms/modules/quarot/cuda/tensor_16_had
# ./fms/modules/quarot/cuda/manual_buffer_test
# ./fms/modules/quarot/cuda/cp_async_test
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:/home/intern/our-fms /home/intern/miniconda3/envs/test/bin/python /home/intern/our-fms/fms/modules/quarot/fast_had_trans.py
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:/home/intern/our-fms /home/intern/miniconda3/envs/test/bin/python /home/intern/our-fms/scripts/inference.py --device_type cuda --model_path /data/intern/llama2-7b/ --tokenizer /data/intern/llama2-7b/tokenizer.model --variant 7b --model_source hf
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:/home/intern/our-fms /home/intern/miniconda3/envs/myenv/bin/python /home/intern/our-fms/scripts/inference.py --device_type cuda --model_path /data/intern/llama2-7b/ --tokenizer /data/intern/llama2-7b/tokenizer.model --variant 7b --model_source hf --compile