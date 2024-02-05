export CUDA_VISIBLE_DEVICES=3,4,5
#W_ELEM_FORAMT="fp6_e3m2"
W_ELEM_FORAMT="fp6_e2m3"
A_ELEM_FORAMT=${W_ELEM_FORAMT}
MX_SPECS="--no_quantize_backprop --w_elem_format ${W_ELEM_FORAMT} --a_elem_format ${A_ELEM_FORAMT} --scale_bits 8 --block_size 32 --bfloat 16 --round even --custom_cuda"
#python -u main.py --model deit_small_patch16_224 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port 12345 main.py --model deit_small_patch16_224 \
	--data-path /group/modelzoo/test_dataset/Imagenet \
       	--batch-size 128 \
	--epochs 10 \
	--resume ./deit_small_e2m3/best_checkpoint.pth \
        --lr 1e-5 \
	--warmup-lr 1e-6 \
	--warmup-epochs 3 \
	--world_size 4 \
	--dist-eval \
	--distributed \
	--output_dir deit_small_e2m3_1 ${MX_SPECS}
