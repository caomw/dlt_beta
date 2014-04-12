%function matcaffe_init(model_def_file, model_file, use_gpu, batch_size)
% function matcaffe_init(model_def_file, model_file, use_gpu [, batch_size])
% 
use_gpu = true;
fast = false;

if(fast)
	model_def_file = '/mnt/neocortex/scratch/tsechiw/tracking/dlt_beta/caffe/cifar100_deploy.prototxt';
	model_file = '/mnt/neocortex/scratch/tsechiw/caffe/build/caffe_cifar10_ft_iter_70000';
else
	model_def_file = '/mnt/neocortex/scratch/tsechiw/tracking/dlt_beta/caffe/vocnet_deploy.prototxt';
	model_file = '/mnt/neocortex/scratch/tsechiw/caffe/build/finetune-by-voc2007/caffe_vocnet_train_iter_150000';
end


global caffe_batch_size
tmpDir = '/tmp/';

caffe('set_device',1);

if ~exist('batch_size','var')
	batch_size = 250;
end

if ~exist( model_def_file , 'file' )
    error('matcaffe_init : I cannot find model_def_file');
end

if ~exist( model_file , 'file' )
    error('matcaffe_init : I cannot find model_file');
end

% create a temp model_def_file with assigned batch_size
caffe_batch_size = batch_size;


modified_def_file = fullfile( tmpDir, ...
    [ 'caffe_deploy_' datestr(now,30) '-' sprintf( '%07d', randi(1e9-1) ) ] );

fidIn  = fopen( model_def_file, 'r' );
fidOut = fopen( modified_def_file, 'w' );

is_modified = 0;
tline = fgetl(fidIn);
while ischar(tline)
    if ~is_modified && strncmp( tline, 'input_dim:', length('input_dim:') )
        fprintf( fidOut, 'input_dim: %d\n', batch_size );
        is_modified = 1;
    else
        fprintf( fidOut, '%s\n', tline );
    end
    tline = fgetl(fidIn);
end

fclose( fidIn );
fclose( fidOut );


% init caffe network (spews logging info)
caffe('init', modified_def_file, model_file);

delete( modified_def_file );

% set to use GPU or CPU
if exist('use_gpu', 'var') && use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end

% put into test mode
caffe('set_phase_test');
