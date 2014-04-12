function [] = viewResult(title)
	dataPath = '../Dataset/';
	addpath('drawUtility');
	
	fullPath = [dataPath, title, '/img/'];
	%fullPath = [dataPath, '/' 'img/'];
	d = dir([fullPath, '*.jpg']);
	if size(d, 1) == 0
		d = dir([fullPath, '*.png']);
	end
	if size(d, 1) == 0
		d = dir([fullPath, '*.bmp']);
	end
	im = imread([fullPath, d(1).name]);

	% Load data
	disp('Loading data...');
	data = zeros(size(im, 1), size(im, 2), 3, size(d, 1));
	for i = 1 : size(d, 1)
		data(:, :, :, i) = imread([fullPath, d(i).name]);
	end
	load([title '_dlt.mat']);
	for i = 1:size(d, 1)
		frame = data(:,:,:,i);
		imshow(frame);
		hold on
		drawTrackRst(frame, savedRes(i,:));
	end
end