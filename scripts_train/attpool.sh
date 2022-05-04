export CXX="g++"
python train.py \
	--conv_enc \
	--att_pool \
	--batchSize $BSIZE \
	--nThreads $NWK \
	--name comod_vgg_l1_conv \
	--train_image_dir ./datasets/places/places2 \
	--train_image_list ./datasets/places/places2_standard_challenge.txt \
	--train_image_postfix '' \
	--val_image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
	--val_image_list ./datasets/places2sample1k_val/files.txt \
	--val_mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
	--load_size 512 \
	--crop_size 256 \
	--z_dim 512 \
	--validation_freq 10000 \
	--niter 50 \
	--dataset_mode trainimage \
	--trainer stylegan2 \
	--dataset_mode_train trainimage \
	--dataset_mode_val valimage \
	--model comod \
	--netG comodgan \
	--netD comodgan \
	--preprocess_mode scale_shortside_and_crop \
	$EXTRA
