W_ELEM_FORAMT="fp6_e3m2"
A_ELEM_FORAMT=${W_ELEM_FORAMT}
MX_SPECS="--no_quantize_backprop --w_elem_format ${W_ELEM_FORAMT} --a_elem_format ${A_ELEM_FORAMT} --scale_bits 8 --block_size 32 --bfloat 16 --round even --custom_cuda"
#python -u main.py --no-mx --model deit_small_patch16_224 \
python -u main.py --resume ./deit_tiny_e3m2/best_checkpoint.pth --model deit_tiny_patch16_224 \
	--data-path /group/modelzoo/test_dataset/Imagenet \
       	--batch-size 200 \
	--eval ${MX_SPECS}
