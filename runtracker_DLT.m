%% Copyright (C) Naiyan Wang and Dit-Yan Yeung.
%% Learning A Deep Compact Image Representation for Visual Tracking. (NIPS2013')
%% All rights reserved.

clc; clear; close;
addpath('affineUtility');
addpath('drawUtility');
addpath('imageUtility');
addpath('NN');
addpath('caffe');

% initialize variables
global DEBUG;
DEBUG = true;

matcaffe_init;
trackparam_DLT;
rand('state',0);  randn('state',0);
frame = data(:,:,:,1);
 
if ~exist('opt','var')  
opt = [];  end

param.est = param0;
savedRes = [];

hold on
drawTrackRst(frame, param.est);

for f = 1:size(data,4)  
	frame = data(:,:,:,f);
	p_prev = p;

	% do tracking
	estwarp_condens_DLT;
	p = param.est;
	opt.affsig = [p(3)/2, p(4)/2, p(3)*0.1, p(4)*0.1];
	opt.motion = [p(1)-p_prev(1), p(2)-p_prev(2)];
	savedRes = [savedRes; p];

	drawTrackRst(frame, p);
	disp(sprintf('Process %d/%d', f, size(data,4)));
end

save([title '_cifar_dlt'], 'savedRes');
%fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);

