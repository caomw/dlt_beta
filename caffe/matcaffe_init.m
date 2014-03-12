%function matcaffe_init(model_def_file, model_file, use_gpu, batch_size)
% function matcaffe_init(model_def_file, model_file, use_gpu [, batch_size])
% 
use_gpu = true;

model_def_file = './caffe/vocnet_deploy.prototxt';
model_file = '../../caffe/build/caffe_vocnet_train_iter_130000';

global caffe_batch_size
tmpDir = '/tmp/';

caffe('set_device',0);

if ~exist('batch_size','var')
	batch_size = 100;
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
