export CXX="g++"
python train.py \
	--batchSize 2 \
	--nThreads 2 \
	--name comod_places \
	--train_image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
	--train_image_list ./datasets/places2sample1k_val/files.txt \
	--train_image_postfix '.jpg' \
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
	--no_l1_loss \
	--no_vgg_loss \
	--preprocess_mode scale_shortside_and_crop \
	$EXTRA
