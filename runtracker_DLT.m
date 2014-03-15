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
 
if ~exist('opt','var')  opt = [];  end
if ~isfield(opt,'minopt')
  opt.minopt = optimset; opt.minopt.MaxIter = 25; opt.minopt.Display='off';
end

param.est = param0;
savedRes = [];

hold on
drawTrackRst(frame, affparam2geom(param.est)');

for f = 1:size(data,4)  
	frame = data(:,:,:,f);
	p_prev = p;

	% do tracking
	param = estwarp_condens_DLT(frame, param, opt);

	res = affparam2geom(param.est);
	p(1) = round(res(1));
	p(2) = round(res(2)); 
	p(3) = res(3) * opt.tmplsize(2);
	p(4) = res(5) * (opt.tmplsize(1) / opt.tmplsize(2)) * p(3);
	p(5) = res(4);
	p(3) = round(p(3));
	p(4) = round(p(4));
	opt.motion = [p(1)-p_prev(1), p(2)-p_prev(2)];
	savedRes = [savedRes; p];

	drawTrackRst(frame, affparam2geom(param.est));
	disp(sprintf('Process %d/%d', f, size(data,4)));
end

save([title '_dlt'], 'savedRes');
fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);

