python test.py \
	--mixing 0 \
	--batchSize 1 \
	--nThreads 1 \
	--name comod-ffhq-512 \
	--dataset_mode testimage \
	--image_dir ./ffhq_debug/images \
	--mask_dir ./ffhq_debug/masks \
        --output_dir ./ffhq_debug \
	--load_size 512 \
	--crop_size 512 \
	--z_dim 512 \
	--model comod \
	--netG comodgan \
        --which_epoch co-mod-gan-ffhq-9-025000 \
	${EXTRA} \
